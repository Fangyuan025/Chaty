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
      // Only the shared context carries a wedged flag — the speech queue owns
      // its own and is rebuilt with the session, so marking it here would
      // churn a context that was never the problem.
      if (!ok && ctx === playbackContext) playbackWedged = true;
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
/** How far ahead of the clock a clip may be scheduled. Enough that `start()`
 *  is never handed a time already in the past, small enough not to be heard. */
const SCHEDULE_LEAD = 0.02;

export class SpeechQueue {
  readonly analyser: AnalyserNode;
  private ctx: AudioContext;
  /** Context time the next clip begins — the end of everything queued. */
  private nextAt = 0;
  private live = new Set<AudioBufferSourceNode>();
  private timers = new Set<ReturnType<typeof setTimeout>>();
  /** Resolves once every clip so far has been SCHEDULED (not played). */
  private tail: Promise<void> = Promise.resolve();
  /** Resolves when the last clip scheduled so far stops sounding. */
  private lastEnded: Promise<void> = Promise.resolve();
  private resumed: Promise<boolean> | null = null;
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
    if (this.stopped || samples.length === 0) return;
    this.tail = this.tail.then(() => this.schedule(samples, sampleRate, label));
  }

  /** Resume once for the queue rather than once per clip: the wait belongs
   *  before the first clip, not between every pair of them. */
  private ready(): Promise<boolean> {
    if (!this.resumed) this.resumed = resumeWithin(this.ctx);
    return this.resumed;
  }

  private async schedule(samples: Float32Array, sampleRate: number, label: string) {
    if (this.stopped) return;
    if (!(await this.ready())) {
      console.warn(`speech queue output did not resume (state: ${this.ctx.state})`);
      return;
    }
    if (this.stopped) return;
    try {
      const buffer = this.ctx.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples, 0);
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.analyser);

      // Clips are stitched on the context clock, so the audio thread runs one
      // into the next sample-accurately. Waiting for `onended` to come back to
      // the main thread and only then building the next clip put a whole event
      // loop between every sentence — audible as choppy speech whenever the
      // main thread was busy, which during a spoken reply it always is.
      const startAt = Math.max(this.ctx.currentTime + SCHEDULE_LEAD, this.nextAt);
      this.live.add(src);
      this.lastEnded = new Promise<void>((resolve) => {
        src.onended = () => {
          this.live.delete(src);
          resolve();
        };
      });
      src.start(startAt);
      this.nextAt = startAt + buffer.duration;

      if (this.onClipStart) {
        const delay = Math.max(0, (startAt - this.ctx.currentTime) * 1000);
        const timer = setTimeout(() => {
          this.timers.delete(timer);
          if (!this.stopped) this.onClipStart?.(label);
        }, delay);
        this.timers.add(timer);
      }
    } catch {
      /* a clip that cannot be built is skipped rather than stalling the rest */
    }
  }

  /** Resolves once every queued clip has finished playing. */
  async whenIdle(): Promise<void> {
    // Clips can still arrive while waiting — settle only when nothing moved.
    for (;;) {
      const scheduled = this.tail;
      const ended = this.lastEnded;
      await scheduled;
      await ended;
      if (scheduled === this.tail && ended === this.lastEnded) return;
    }
  }

  get isStopped() {
    return this.stopped;
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    for (const timer of this.timers) clearTimeout(timer);
    this.timers.clear();
    // Everything scheduled ahead of the clock has to be stopped, not just
    // whatever happens to be sounding right now.
    for (const src of this.live) {
      try {
        src.stop();
      } catch {
        /* not started, or already stopped */
      }
    }
    this.live.clear();
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
