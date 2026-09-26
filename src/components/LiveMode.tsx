import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import {
  cancelGeneration,
  generate,
  synthesize,
  hasHan,
  transcribe,
  type ChatMessage,
} from "../lib/ipc";
import {
  decodeAudio,
  encodeAudio,
  readLevel,
  SpeechQueue,
  startRecording,
  type Recorder,
} from "../lib/audio";
import { answerOnly, cutSentences, forSpeech } from "../lib/voiceText";
import { isVoiceDownloadCancelled } from "../lib/voiceError";

type Status = "listening" | "thinking" | "speaking";

type Rgb = [number, number, number];

/** The glow's hues per state, one per light: cool and quiet while it listens,
 *  indigo while it thinks, bright blue-violet while it speaks. The juniper in
 *  the listening set is the app's accent. */
const PALETTES: Record<Status, Rgb[]> = {
  listening: [
    [43, 179, 160],
    [59, 130, 246],
    [63, 138, 101],
    [56, 189, 248],
    [45, 212, 191],
  ],
  thinking: [
    [99, 102, 241],
    [139, 92, 246],
    [59, 130, 246],
    [20, 184, 166],
    [124, 58, 237],
  ],
  speaking: [
    [59, 130, 246],
    [34, 211, 238],
    [167, 139, 250],
    [99, 102, 241],
    [56, 189, 248],
  ],
};

/**
 * Gemini-style hands-free voice conversation: an animated orb reacting to the
 * live audio level, looping listen → transcribe → LLM → speak → listen.
 */
export function LiveMode({
  onClose,
  preamble,
  initialHistory,
  onTurn,
  appendNoThink,
  forceNoThink,
  voiceSid,
  voiceSidZh,
  voiceSpeed,
  chineseVoice,
}: {
  onClose: () => void;
  preamble: string;
  initialHistory: ChatMessage[];
  onTurn: (userText: string, assistantText: string) => void;
  appendNoThink: boolean;
  /** Switch-less reasoning models (Qwen3.5+): disable thinking via the backend. */
  forceNoThink: boolean;
  voiceSid: number;
  /** Speaker for the Chinese voice — its own list, not an index into the
   *  English one. */
  voiceSidZh: number;
  voiceSpeed: number;
  chineseVoice: boolean;
}) {
  const { t } = useI18n();
  const [status, setStatus] = useState<Status>("listening");
  const [caption, setCaption] = useState("");
  const [error, setError] = useState("");

  const activeRef = useRef(true);
  const statusRef = useRef<Status>("listening");
  const levelRef = useRef<(() => number) | null>(null);
  const recorderRef = useRef<Recorder | null>(null);
  const speechRef = useRef<SpeechQueue | null>(null);
  const cancelCaptureRef = useRef<(() => void) | null>(null);
  const messagesRef = useRef<ChatMessage[]>([...initialHistory]);
  const onTurnRef = useRef(onTurn);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    onTurnRef.current = onTurn;
  });

  // Keep the (scrollable, height-capped) transcript pinned to the latest line.
  useEffect(() => {
    const el = captionRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [caption]);

  const setBoth = (s: Status) => {
    statusRef.current = s;
    setStatus(s);
  };

  // ---- the glow ----
  // A band of soft light along the bottom edge, in the manner of Gemini
  // Live: a handful of coloured lights drifting under a heavy blur, rising
  // and brightening with the voice (the microphone while it listens, the
  // reply while it speaks). The canvas is drawn at a quarter of the screen's
  // resolution — the blur hides it, and a frame costs next to nothing.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    const fit = () => {
      canvas.width = Math.max(64, Math.round(canvas.clientWidth / 4));
      canvas.height = Math.max(32, Math.round(canvas.clientHeight / 4));
    };
    fit();
    window.addEventListener("resize", fit);

    const seeds = [0.3, 1.7, 3.1, 4.4, 5.6];
    const colors: Rgb[] = PALETTES.listening.map((c) => [...c] as Rgb);
    let raf = 0;
    let smooth = 0;
    let lift = 0.35;
    let shown = 0;
    const render = () => {
      let raw = 0;
      try {
        raw = levelRef.current?.() ?? 0;
      } catch {
        raw = 0; // the source may have just closed between turns
      }
      smooth += (raw - smooth) * 0.18;
      const st = statusRef.current;
      const t = (performance.now() / 1000) * (calm ? 0.25 : 1);
      const level = Math.min(1, smooth * 3);
      // Colours ease towards the state's set rather than jumping.
      const target = PALETTES[st];
      for (let i = 0; i < colors.length; i++) {
        for (let k = 0; k < 3; k++) colors[i][k] += (target[i][k] - colors[i][k]) * 0.04;
      }
      // How far up the band reaches: it swells with the voice but stays
      // below the words in the middle of the screen.
      const wantLift =
        st === "speaking" ? 0.42 + level * 0.3 : st === "thinking" ? 0.36 + 0.05 * Math.sin(t * 2.2) : 0.28 + level * 0.28;
      lift += (wantLift - lift) * 0.08;
      shown = Math.min(1, shown + 0.03); // fade in on open

      const w = canvas.width;
      const h = canvas.height;
      ctx.globalCompositeOperation = "source-over";
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";
      const n = colors.length;
      for (let i = 0; i < n; i++) {
        const p = seeds[i];
        const sweep = st === "thinking" ? Math.sin(t * 0.9 + p) * w * 0.12 : 0;
        const x = w * (0.06 + (0.88 * i) / (n - 1)) + Math.sin(t * (0.23 + 0.07 * i) + p) * w * 0.08 + sweep;
        const y = h * (1.08 - lift * 0.55) + Math.sin(t * 0.5 + p * 2) * h * 0.05;
        const r = h * (0.62 + 0.16 * Math.sin(t * 0.37 + p * 3)) * (0.8 + lift * 0.7);
        const a = shown * (st === "listening" ? 0.4 + level * 0.3 : st === "thinking" ? 0.46 : 0.52 + level * 0.28);
        const [R, G, B] = colors[i].map(Math.round);
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0, `rgba(${R},${G},${B},${a})`);
        g.addColorStop(0.45, `rgba(${R},${G},${B},${a * 0.45})`);
        g.addColorStop(1, `rgba(${R},${G},${B},0)`);
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
      }
      // A faint pale core along the very bottom, strongest while it speaks.
      const core = ctx.createLinearGradient(0, h, 0, h * (1 - lift * 0.5));
      const ca = shown * (0.1 + (st === "speaking" ? 0.12 + level * 0.18 : level * 0.1));
      core.addColorStop(0, `rgba(220,235,255,${ca})`);
      core.addColorStop(1, "rgba(220,235,255,0)");
      ctx.fillStyle = core;
      ctx.fillRect(0, 0, w, h);

      raf = requestAnimationFrame(render);
    };
    render();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", fit);
    };
  }, []);

  // ---- the conversation loop ----
  useEffect(() => {
    activeRef.current = true;
    // Defer startup by one task. React StrictMode intentionally runs
    // effect setup → cleanup → setup in development; starting mic IPC
    // immediately lets the discarded first setup retain the native recorder
    // while the real setup receives "already recording".
    const startTimer = window.setTimeout(() => {
      if (activeRef.current) void loop();
    }, 0);
    return () => {
      window.clearTimeout(startTimer);
      activeRef.current = false;
      cancelCaptureRef.current?.();
      void recorderRef.current?.cancel().catch(() => {});
      recorderRef.current = null;
      speechRef.current?.stop();
      speechRef.current = null;
      cancelGeneration().catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Record one utterance, auto-stopping on silence (VAD). */
  function captureUtterance(): Promise<{ samples: Float32Array; sampleRate: number } | null> {
    return new Promise((resolve) => {
      let done = false;
      const finish = async () => {
        const rec = recorderRef.current;
        if (done || !rec) return;
        done = true;
        recorderRef.current = null;
        try {
          resolve(await rec.stop());
        } catch {
          resolve(null);
        }
      };
      cancelCaptureRef.current = () => {
        if (done) return;
        done = true;
        void recorderRef.current?.cancel().catch(() => {});
        recorderRef.current = null;
        resolve(null);
      };
      startRecording({ onAutoStop: () => void finish(), silenceMs: 1000 })
        .then(async (rec) => {
          if (!activeRef.current) {
            await rec.cancel().catch(() => {});
            resolve(null);
            return;
          }
          recorderRef.current = rec;
          levelRef.current = rec.level;
          setBoth("listening");
        })
        .catch((e) => {
          setError(String(e));
          resolve(null);
        });
    });
  }

  async function loop() {
    while (activeRef.current) {
      const cap = await captureUtterance();
      if (!activeRef.current || !cap) break;
      // Ignore blips that are too short to be speech.
      if (cap.samples.length < cap.sampleRate * 0.35) continue;

      setBoth("thinking");
      setCaption("");
      let userText = "";
      try {
        userText = (
          await transcribe(encodeAudio(cap.samples), cap.sampleRate, chineseVoice)
        ).trim();
      } catch (e) {
        setError(String(e));
      }
      if (!activeRef.current) break;
      if (!userText) {
        setBoth("listening");
        continue;
      }
      setCaption(userText);
      messagesRef.current.push({ role: "user", content: userText });
      await respond(userText);
    }
  }

  async function respond(userText: string) {
    let acc = "";
    let spokenLen = 0;
    let synthChain: Promise<void> = Promise.resolve();
    let started = false;

    // Show only the sentence currently being spoken (big and centered); the
    // full transcript is saved to the conversation for later review.
    const speech = new SpeechQueue((label) => {
      if (!activeRef.current) return;
      if (!started) {
        started = true;
        setBoth("speaking");
      }
      setCaption(label);
    });
    speechRef.current = speech;
    const speechBuf = new Uint8Array(1024);
    levelRef.current = () => readLevel(speech.analyser, speechBuf);

    const enqueue = (raw: string) => {
      const clean = forSpeech(raw);
      if (!clean) return;
      synthChain = synthChain.then(async () => {
        if (speech.isStopped) return;
        try {
          const { audio, sampleRate } = await synthesize(
            clean,
            voiceSpeed,
            voiceSid,
            chineseVoice,
            voiceSidZh,
            hasHan(answerOnly(acc)),
          );
          if (!speech.isStopped) speech.enqueue(decodeAudio(audio), sampleRate, clean);
        } catch (e) {
          if (!isVoiceDownloadCancelled(e)) setError(String(e));
        }
      });
    };
    const pump = (final: boolean) => {
      const ans = answerOnly(acc);
      let pending = ans.slice(spokenLen);
      if (final) {
        spokenLen = ans.length;
      } else {
        const [d] = cutSentences(pending);
        if (!d) return;
        pending = d;
        spokenLen += d.length;
      }
      enqueue(pending);
    };

    const messages: ChatMessage[] = [
      { role: "system", content: preamble },
      ...messagesRef.current,
    ];
    // Disable the model's reasoning in live mode for snappy spoken turns — but
    // only if the model actually supports `/no_think` (else it's just noise).
    if (appendNoThink) {
      messages[messages.length - 1] = {
        ...messages[messages.length - 1],
        content: `${messages[messages.length - 1].content}\n/no_think`,
      };
    }

    try {
      await generate(
        {
          messages,
          params: {
            temperature: 0.6,
            topP: 0.9,
            maxTokens: 400,
            think: forceNoThink ? false : undefined,
          },
        },
        (ev) => {
          if (ev.type === "token") {
            acc += ev.text;
            pump(false); // synthesize as sentences complete; transcript shows on playback
          }
        },
      );
    } catch (e) {
      setError(String(e));
    }
    pump(true);

    const answer = answerOnly(acc).trim();
    messagesRef.current.push({ role: "assistant", content: answer });
    // Record the live turn into the conversation history.
    if (userText && answer) onTurnRef.current(userText, answer);

    await synthChain;
    await speech.whenIdle();
    // Free the AudioContext (browsers cap concurrent contexts ~6, so a long
    // live session would otherwise throw after a handful of turns).
    speech.stop();
    if (speechRef.current === speech) speechRef.current = null;
    if (activeRef.current) {
      setBoth("listening");
      setCaption("");
    }
  }

  const statusText =
    status === "listening" ? t("liveListening") : status === "thinking" ? t("liveThinking") : t("liveSpeaking");

  return createPortal(
    <div className={`live-overlay live-${status}`}>
      <canvas ref={canvasRef} className="live-glow" aria-hidden="true" />
      <div className="live-top">
        <span className="live-label">{t("liveStart")}</span>
        <span className="live-status">{statusText}</span>
      </div>
      <div className="live-stage">
        {caption ? (
          <div className="live-caption" ref={captionRef}>
            {caption}
          </div>
        ) : (
          <div className="live-idle">{statusText}</div>
        )}
        {error && <div className="live-error">{error}</div>}
      </div>
      <div className="live-controls">
        <button className="live-end" onClick={onClose} title={t("liveExit")} aria-label={t("liveExit")}>
          <Icon name="x" size={20} strokeWidth={2} />
        </button>
        <span className="live-end-label">{t("liveEnd")}</span>
      </div>
    </div>,
    document.body,
  );
}
