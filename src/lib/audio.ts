// Web Audio helpers for voice: capture, base64 (de)serialization, playback.

import { invoke } from "@tauri-apps/api/core";
import { platform } from "@tauri-apps/plugin-os";

/** Encode Float32 PCM as base64 of its little-endian bytes. */
export function encodeAudio(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

/** Decode base64 of little-endian f32 bytes back into Float32 PCM. */
export function decodeAudio(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export interface Recorder {
  /** Current input level (~0..1 RMS-ish) for level meters / orb animation. */
  level: () => number;
  /** Stop and return the captured mono PCM. */
  stop: () => Promise<{ samples: Float32Array; sampleRate: number }>;
  /** Stop and discard. */
  cancel: () => Promise<void>;
}

export interface RecordOptions {
  /** Fired once when the user goes silent after having spoken (auto-stop). */
  onAutoStop?: () => void;
  /** Silence duration (ms) that ends the recording. Default 1100. */
  silenceMs?: number;
}

/** Start capturing microphone audio as mono Float32 PCM. */
export async function startRecording(opts?: RecordOptions): Promise<Recorder> {
  // macOS: capture natively (CoreAudio via Rust). WKWebView never exposes
  // capture devices to embedded apps — getUserMedia fails with "0 devices"
  // regardless of TCC, entitlements, or WebKit preference flags.
  let mac = false;
  try {
    mac = platform() === "macos";
  } catch {
    /* non-Tauri context */
  }
  if (mac) return startNativeRecording(opts);
  // Plain `audio: true` — WKWebView (macOS) throws OverconstrainedError for
  // audio-processing constraint keys (echoCancellation & co.), and engines
  // apply sensible processing defaults anyway. Surface a readable error with
  // device diagnostics if capture still fails.
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    const name = (e as { name?: string } | null)?.name ?? "";
    const msg = (e as { message?: string } | null)?.message ?? String(e);
    let inputs = -1;
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      inputs = devs.filter((d) => d.kind === "audioinput").length;
    } catch {
      /* diagnostics only */
    }
    throw new Error(`${name || "Error"}: ${msg} (audio inputs visible: ${inputs})`);
  }
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);

  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(ctx.destination);
  const sampleRate = ctx.sampleRate;

  // Voice-activity detection: once the user has spoken, end the recording after
  // a stretch of silence (Gemini-style hands-free turn-taking).
  let vadTimer = 0;
  if (opts?.onAutoStop) {
    const buf = new Uint8Array(analyser.fftSize);
    let speechStarted = false;
    let lastVoice = performance.now();
    let fired = false;
    vadTimer = window.setInterval(() => {
      const level = readLevel(analyser, buf);
      const now = performance.now();
      if (level > 0.05) {
        speechStarted = true;
        lastVoice = now;
      }
      if (!fired && speechStarted && now - lastVoice > (opts.silenceMs ?? 1100)) {
        fired = true;
        window.clearInterval(vadTimer);
        opts.onAutoStop?.();
      }
    }, 120);
  }

  const teardown = () => {
    if (vadTimer) window.clearInterval(vadTimer);
    processor.onaudioprocess = null;
    processor.disconnect();
    analyser.disconnect();
    source.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    ctx.close().catch(() => {});
  };

  const levelBuf = new Uint8Array(analyser.fftSize);
  return {
    level: () => {
      try {
        return readLevel(analyser, levelBuf);
      } catch {
        return 0;
      }
    },
    cancel: async () => teardown(),
    stop: async () => {
      teardown();
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const out = new Float32Array(total);
      let off = 0;
      for (const c of chunks) {
        out.set(c, off);
        off += c.length;
      }
      return { samples: out, sampleRate };
    },
  };
}

/** macOS-native recorder: Rust/cpal capture, level polled over IPC. */
async function startNativeRecording(opts?: RecordOptions): Promise<Recorder> {
  const sampleRate = await invoke<number>("mic_start");
  let last = 0;
  let speechStarted = false;
  let lastVoice = performance.now();
  let fired = false;
  let stopped = false;
  const poll = window.setInterval(() => {
    invoke<number>("mic_level")
      .then((lv) => {
        last = lv;
        if (!opts?.onAutoStop || fired) return;
        const now = performance.now();
        if (lv > 0.04) {
          speechStarted = true;
          lastVoice = now;
        }
        if (speechStarted && now - lastVoice > (opts.silenceMs ?? 1100)) {
          fired = true;
          opts.onAutoStop?.();
        }
      })
      .catch(() => {});
  }, 100);
  const teardown = () => window.clearInterval(poll);
  return {
    level: () => last,
    cancel: async () => {
      if (stopped) return;
      stopped = true;
      teardown();
      await invoke("mic_cancel");
    },
    stop: async () => {
      teardown();
      if (stopped) return { samples: new Float32Array(0), sampleRate };
      stopped = true;
      const res = await invoke<{ samples: string; sampleRate: number }>("mic_stop");
      return { samples: decodeAudio(res.samples), sampleRate: res.sampleRate };
    },
  };
}

export interface Playback {
  /** Analyser of the output (for the speaking animation). */
  analyser: AnalyserNode;
  /** Resolves when playback finishes. */
  done: Promise<void>;
  stop: () => void;
}

// WKWebView applies the same user-activation rules as Safari: an AudioContext
// created only after an async TTS IPC call may start suspended because the
// original click is no longer considered active. Keep one output context and
// unlock it on the first pointer/keyboard gesture, before synthesis begins.
let playbackContext: AudioContext | null = null;
/** Set when an output context stopped answering, so the next utterance builds
 *  a fresh one instead of speaking into the dead one again. */
let playbackWedged = false;

/// WebKit parks a context as `interrupted` — an audio route change, the
/// machine sleeping, another app taking the device — and `resume()` on one of
/// those can reject or simply never settle. Replacing only a `closed` context
/// left the wedged one in place for every later utterance, which is why speech
/// that stopped stayed stopped until the app was restarted.
function sharedPlaybackContext(): AudioContext {
  const state = playbackContext?.state as string | undefined;
  if (!playbackContext || playbackWedged || state === "closed" || state === "interrupted") {
    try {
      void playbackContext?.close();
    } catch {
      /* already gone */
    }
    playbackContext = new AudioContext();
    playbackWedged = false;
  }
  return playbackContext;
}

/** How long `resume()` may take before the context counts as wedged. Long
 *  enough that a busy machine is not mistaken for a dead device. */
const RESUME_TIMEOUT_MS = 2_000;

/** Resolves false when the context did not come back — rejected, or never
 *  answered at all, which is the failure that has no error to catch. */
function resumeWithin(ctx: AudioContext): Promise<boolean> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      if (!ok) playbackWedged = true;
      resolve(ok);
    };
    const timer = setTimeout(() => finish(false), RESUME_TIMEOUT_MS);
    ctx.resume().then(
      () => {
        clearTimeout(timer);
        finish(ctx.state === "running");
      },
      () => {
        clearTimeout(timer);
        finish(false);
      },
    );
  });
}

/** Unlock Web Audio while a real user gesture is still active. */
export function primeAudioPlayback(): void {
  if (typeof AudioContext === "undefined") return;
  const ctx = sharedPlaybackContext();
  if (ctx.state === "suspended") void ctx.resume().catch(() => {});
}

if (typeof window !== "undefined" && typeof AudioContext !== "undefined") {
  const unlock = () => primeAudioPlayback();
  window.addEventListener("pointerdown", unlock, { capture: true, once: true });
  window.addEventListener("keydown", unlock, { capture: true, once: true });
}

/** Play Float32 PCM; returns an analyser + a promise that resolves when done. */
export function playAudio(samples: Float32Array, sampleRate: number): Playback {
  const ctx = sharedPlaybackContext();
  const buffer = ctx.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  analyser.connect(ctx.destination);

  let stopped = false;
  let finished = false;
  let resolveDone!: () => void;
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve;
  });
  const finish = () => {
    if (finished) return;
    finished = true;
    try {
      source.disconnect();
      analyser.disconnect();
    } catch {
      /* already disconnected */
    }
    resolveDone();
  };
  source.onended = finish;
  void resumeWithin(ctx).then((ok) => {
    if (stopped) {
      finish();
      return;
    }
    if (!ok) {
      // Nothing to catch here — the context simply stopped answering. Say so,
      // and let the next utterance start over on a new one.
      console.warn(`audio output did not resume (state: ${ctx.state}); replacing it`);
      finish();
      return;
    }
    source.start();
  });

  return {
    analyser,
    done,
    stop: () => {
      if (stopped) return;
      stopped = true;
      try {
        source.stop();
      } catch {
        // Not started yet, or already stopped.
        finish();
      }
    },
  };
}

/**
 * A gap-less playback queue: clips are played back-to-back in enqueue order,
 * so a reply can be synthesized sentence-by-sentence and start playing before
 * the whole answer is generated. A single analyser feeds the speaking orb.
 */
export class SpeechQueue {
  readonly analyser: AnalyserNode;
  private ctx: AudioContext;
  private tail: Promise<void> = Promise.resolve();
  private current: AudioBufferSourceNode | null = null;
  private stopped = false;
  private onClipStart?: (label: string) => void;

  /** `onClipStart(label)` fires the moment each queued clip begins playing —
   *  used to reveal a transcript in sync with the audio. */
  constructor(onClipStart?: (label: string) => void) {
    this.onClipStart = onClipStart;
    this.ctx = new AudioContext();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.connect(this.ctx.destination);
  }

  /** Queue a clip for playback after everything already queued. */
  enqueue(samples: Float32Array, sampleRate: number, label = "") {
    this.tail = this.tail.then(() => this.playClip(samples, sampleRate, label));
  }

  private playClip(samples: Float32Array, sampleRate: number, label: string): Promise<void> {
    if (this.stopped || samples.length === 0) return Promise.resolve();
    return new Promise<void>((resolve) => {
      // The resume has to be waited on. Starting a source on a context that
      // never came back schedules a clip that never ends, and since each clip
      // gates the next one, the whole queue stops with it.
      void resumeWithin(this.ctx).then((ok) => {
        if (this.stopped) {
          resolve();
          return;
        }
        if (!ok) {
          console.warn(`speech queue output did not resume (state: ${this.ctx.state})`);
          resolve();
          return;
        }
        this.startClip(samples, sampleRate, label, resolve);
      });
    });
  }

  private startClip(
    samples: Float32Array,
    sampleRate: number,
    label: string,
    resolve: () => void,
  ): void {
    try {
      const buffer = this.ctx.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples, 0);
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.analyser);
      this.current = src;
      src.onended = () => {
        this.current = null;
        resolve();
      };
      this.onClipStart?.(label);
      src.start();
    } catch {
      resolve();
    }
  }

  /** Resolves once every queued clip has finished playing. */
  async whenIdle(): Promise<void> {
    await this.tail;
  }

  get isStopped() {
    return this.stopped;
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    try {
      this.current?.stop();
    } catch {
      /* already stopped */
    }
    this.current = null;
    this.ctx.close().catch(() => {});
  }
}

/** RMS amplitude (0..1) from an analyser — for reactive animations. */
export function readLevel(analyser: AnalyserNode, buf: Uint8Array): number {
  analyser.getByteTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) {
    const v = (buf[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / buf.length);
}
