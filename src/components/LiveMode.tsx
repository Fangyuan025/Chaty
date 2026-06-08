import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import {
  cancelGeneration,
  generate,
  synthesize,
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

type Status = "listening" | "thinking" | "speaking";

const COLORS: Record<Status, [number, number, number]> = {
  listening: [25, 195, 125],
  thinking: [240, 178, 50],
  speaking: [74, 163, 255],
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
}: {
  onClose: () => void;
  preamble: string;
  initialHistory: ChatMessage[];
  onTurn: (userText: string, assistantText: string) => void;
  appendNoThink: boolean;
  /** Switch-less reasoning models (Qwen3.5+): disable thinking via the backend. */
  forceNoThink: boolean;
}) {
  const { t } = useI18n();
  const [status, setStatus] = useState<Status>("listening");
  const [caption, setCaption] = useState("");
  const [error, setError] = useState("");

  const activeRef = useRef(true);
  const statusRef = useRef<Status>("listening");
  const analyserRef = useRef<AnalyserNode | null>(null);
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

  // ---- the orb animation ----
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const size = 260;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const buf = new Uint8Array(1024);
    let raf = 0;
    let smooth = 0;
    const render = () => {
      const a = analyserRef.current;
      let raw = 0;
      if (a) {
        try {
          raw = readLevel(a, buf);
        } catch {
          raw = 0; // analyser's context may have just closed between turns
        }
      }
      smooth += (raw - smooth) * 0.2;
      const tNow = performance.now() / 1000;
      const st = statusRef.current;
      const [r, g, b] = COLORS[st];
      const breathe = st === "thinking" ? 0.06 + 0.05 * Math.sin(tNow * 3) : 0;
      const level = Math.min(0.6, smooth * 2.2) + breathe;

      ctx.clearRect(0, 0, size, size);
      const cx = size / 2;
      const cy = size / 2;
      const baseR = 64;
      const radius = baseR * (1 + level);

      // outer glow rings
      for (let i = 3; i >= 1; i--) {
        ctx.beginPath();
        ctx.arc(cx, cy, radius + i * 14 * (0.6 + level), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${0.05 + level * 0.05})`;
        ctx.fill();
      }
      // main orb
      const grad = ctx.createRadialGradient(cx, cy - radius * 0.3, radius * 0.2, cx, cy, radius);
      grad.addColorStop(0, `rgba(${Math.min(r + 60, 255)},${Math.min(g + 60, 255)},${Math.min(b + 60, 255)},1)`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0.92)`);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.shadowBlur = 40 + level * 60;
      ctx.shadowColor = `rgba(${r},${g},${b},0.7)`;
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.shadowBlur = 0;

      raf = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(raf);
  }, []);

  // ---- the conversation loop ----
  useEffect(() => {
    activeRef.current = true;
    void loop();
    return () => {
      activeRef.current = false;
      cancelCaptureRef.current?.();
      recorderRef.current?.cancel();
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
        recorderRef.current?.cancel();
        recorderRef.current = null;
        resolve(null);
      };
      startRecording({ onAutoStop: () => void finish(), silenceMs: 1000 })
        .then((rec) => {
          if (!activeRef.current) {
            rec.cancel();
            resolve(null);
            return;
          }
          recorderRef.current = rec;
          analyserRef.current = rec.analyser;
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
        userText = (await transcribe(encodeAudio(cap.samples), cap.sampleRate)).trim();
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
    analyserRef.current = speech.analyser;

    const enqueue = (raw: string) => {
      const clean = forSpeech(raw);
      if (!clean) return;
      synthChain = synthChain.then(async () => {
        if (speech.isStopped) return;
        try {
          const { audio, sampleRate } = await synthesize(clean);
          if (!speech.isStopped) speech.enqueue(decodeAudio(audio), sampleRate, clean);
        } catch (e) {
          setError(String(e));
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
    <div className="live-overlay">
      <button className="live-close" onClick={onClose} title={t("liveExit")}>
        ×
      </button>
      <div className="live-stage">
        <canvas ref={canvasRef} className="live-orb" style={{ width: 260, height: 260 }} />
        <div className="live-status">{statusText}</div>
        {caption && (
          <div className="live-caption" ref={captionRef}>
            {caption}
          </div>
        )}
        {error && <div className="live-error">{error}</div>}
      </div>
      <button className="live-end" onClick={onClose}>
        {t("liveExit")}
      </button>
    </div>,
    document.body,
  );
}
