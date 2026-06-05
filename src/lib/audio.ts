// Web Audio helpers for voice: capture, base64 (de)serialization, playback.

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
  /** Live analyser of the mic input (for level meters / orb animation). */
  analyser: AnalyserNode;
  /** Stop and return the captured mono PCM. */
  stop: () => Promise<{ samples: Float32Array; sampleRate: number }>;
  /** Stop and discard. */
  cancel: () => void;
}

export interface RecordOptions {
  /** Fired once when the user goes silent after having spoken (auto-stop). */
  onAutoStop?: () => void;
  /** Silence duration (ms) that ends the recording. Default 1100. */
  silenceMs?: number;
}

/** Start capturing microphone audio as mono Float32 PCM. */
export async function startRecording(opts?: RecordOptions): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
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

  return {
    analyser,
    cancel: teardown,
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

export interface Playback {
  /** Analyser of the output (for the speaking animation). */
  analyser: AnalyserNode;
  /** Resolves when playback finishes. */
  done: Promise<void>;
  stop: () => void;
}

/** Play Float32 PCM; returns an analyser + a promise that resolves when done. */
export function playAudio(samples: Float32Array, sampleRate: number): Playback {
  const ctx = new AudioContext();
  const buffer = ctx.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  analyser.connect(ctx.destination);

  let stopped = false;
  const done = new Promise<void>((resolve) => {
    source.onended = () => {
      ctx.close().catch(() => {});
      resolve();
    };
  });
  source.start();

  return {
    analyser,
    done,
    stop: () => {
      if (stopped) return;
      stopped = true;
      try {
        source.stop();
      } catch {
        /* already stopped */
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
      try {
        void this.ctx.resume();
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
    });
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
