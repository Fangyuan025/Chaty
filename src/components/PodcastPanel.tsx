import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import { Icon } from "./Icon";
import { IconMic, IconDownload, IconRefresh, IconPlay, IconStop } from "./icons";
import {
  cancelGeneration,
  exportWavFile,
  generate,
  ragCorpus,
  synthesize,
  type ModelInfo,
} from "../lib/ipc";
import { decodeAudio, encodeAudio, playAudio, type Playback } from "../lib/audio";
import { fmtTime } from "../lib/eta";

// Two hosts with the highest overall-grade voices in kokoro-en-v0_19
// (per the Kokoro VOICES.md): af_bella (A-) ↔ am_michael (C+, the best male in
// this voice pack; am_adam is F+). Alternating, NotebookLM-style.
const VOICE_A = 1; // af_bella · female (A-)
const VOICE_B = 6; // am_michael · male (C+)
const HOST_A = "Maya";
const HOST_B = "Leo";

type Phase = "idle" | "transcript" | "synth" | "ready" | "error";

interface Line {
  speaker: "A" | "B";
  text: string;
}

/** Build the transcript-generation prompt (always English output). */
function transcriptSystemPrompt(): string {
  return [
    `You are the scriptwriter for a NotebookLM-style "deep dive" audio podcast.`,
    `Two hosts discuss the user's source material: ${HOST_A} (warm, curious, asks the questions a smart listener would) and ${HOST_B} (knowledgeable, explains clearly with concrete detail).`,
    `Write an engaging, natural spoken-word conversation in ENGLISH, grounded STRICTLY in the provided sources — never invent facts, names, or numbers that are not supported by them.`,
    `Open with a short hook, cover the most interesting and important ideas, and end with a brief wrap-up.`,
    `Format EVERY line as "${HOST_A}: ..." or "${HOST_B}: ..." on its own line, alternating speakers. Use plain prose only — no markdown, stage directions, bullet points, or headings.`,
    `Aim for 14–22 exchanges.`,
  ].join(" ");
}

/** Parse "Name: text" lines into alternating-speaker turns. */
function parseTranscript(raw: string): Line[] {
  const lines: Line[] = [];
  const a = HOST_A.toLowerCase();
  const b = HOST_B.toLowerCase();
  for (const rawLine of raw.split(/\r?\n/)) {
    const line = rawLine.trim().replace(/^[*_>#\s-]+/, "");
    if (!line) continue;
    const m = line.match(/^\*{0,2}([^:：]{1,24})\*{0,2}\s*[:：]\s*(.+)$/);
    if (m) {
      const name = m[1].trim().toLowerCase();
      const text = m[2].trim();
      if (!text) continue;
      let speaker: "A" | "B" | null = null;
      if (name.includes(a) || name === "a" || /\b(host a|speaker 1|host 1)\b/.test(name)) {
        speaker = "A";
      } else if (name.includes(b) || name === "b" || /\b(host b|speaker 2|host 2)\b/.test(name)) {
        speaker = "B";
      }
      if (speaker) {
        lines.push({ speaker, text });
        continue;
      }
    }
    // Continuation of the previous turn (wrapped line).
    if (lines.length > 0) lines[lines.length - 1].text += " " + line;
  }
  return lines;
}

/** NotebookLM-style deep-dive podcast: KB → English transcript → 2-voice TTS. */
export function PodcastPanel({
  model,
  voiceSpeed,
  onClose,
  onLockChange,
}: {
  model: ModelInfo | null;
  voiceSpeed: number;
  onClose: () => void;
  /** Lock/unlock other LLM features while the podcast is being produced. */
  onLockChange: (locked: boolean) => void;
}) {
  const { t, lang } = useI18n();
  const [phase, setPhase] = useState<Phase>("idle");
  const [transcript, setTranscript] = useState("");
  const [lines, setLines] = useState<Line[]>([]);
  const [done, setDone] = useState(0); // lines synthesized
  const [total, setTotal] = useState(0);
  const [eta, setEta] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [playing, setPlaying] = useState(false);

  const cancelRef = useRef(false);
  const audioRef = useRef<{ samples: Float32Array; sampleRate: number } | null>(null);
  const playbackRef = useRef<Playback | null>(null);

  // Lock LLM features for the whole lifetime of an in-progress generation.
  useEffect(() => {
    const busy = phase === "transcript" || phase === "synth";
    onLockChange(busy);
    return () => onLockChange(false);
  }, [phase, onLockChange]);

  useEffect(() => {
    return () => {
      cancelRef.current = true;
      playbackRef.current?.stop();
    };
  }, []);

  const noThink = model?.supportsThinking && !model.thinkSwitch ? false : undefined;

  async function generateTranscript(): Promise<Line[]> {
    const maxChars = 9000;
    const corpus = await ragCorpus(maxChars);
    const userMsg =
      `Source material for the podcast:\n\n${corpus}\n\n` +
      `Now write the full two-host podcast script in English.` +
      (model?.thinkSwitch ? "\n/no_think" : "");
    let acc = "";
    await generate(
      {
        messages: [
          { role: "system", content: transcriptSystemPrompt() },
          { role: "user", content: userMsg },
        ],
        params: { temperature: 0.7, topP: 0.95, maxTokens: 4096, think: noThink },
      },
      (ev) => {
        if (ev.type === "token") {
          acc += ev.text;
          setTranscript(acc.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, ""));
        }
      },
    );
    const clean = acc.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, "");
    return parseTranscript(clean);
  }

  async function run() {
    if (!model) {
      setError(t("podcastNeedModel"));
      setPhase("error");
      return;
    }
    cancelRef.current = false;
    setError("");
    setDone(0);
    setEta(null);
    setTranscript("");
    setLines([]);
    setPhase("transcript");

    let parsed: Line[];
    try {
      parsed = await generateTranscript();
    } catch (e) {
      if (cancelRef.current) return void resetIdle();
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
      return;
    }
    if (cancelRef.current) return void resetIdle();
    if (parsed.length < 2) {
      setError(t("podcastNoScript"));
      setPhase("error");
      return;
    }
    setLines(parsed);
    setTotal(parsed.length);
    setPhase("synth");

    // Synthesize line-by-line, alternating voices, accumulating PCM.
    const clips: Float32Array[] = [];
    let sampleRate = 24000;
    const startT = performance.now();
    for (let i = 0; i < parsed.length; i++) {
      if (cancelRef.current) return void resetIdle();
      const ln = parsed[i];
      try {
        const { audio, sampleRate: sr } = await synthesize(
          ln.text,
          voiceSpeed,
          ln.speaker === "A" ? VOICE_A : VOICE_B,
        );
        sampleRate = sr;
        clips.push(decodeAudio(audio));
        // Inter-turn pause: longer when the speaker changes.
        const changed = i + 1 < parsed.length && parsed[i + 1].speaker !== ln.speaker;
        clips.push(new Float32Array(Math.round(sr * (changed ? 0.35 : 0.12))));
      } catch (e) {
        if (cancelRef.current) return void resetIdle();
        setError(e instanceof Error ? e.message : String(e));
        setPhase("error");
        return;
      }
      const finished = i + 1;
      setDone(finished);
      const elapsed = (performance.now() - startT) / 1000;
      const perLine = elapsed / finished;
      setEta(perLine * (parsed.length - finished));
    }

    if (cancelRef.current) return void resetIdle();
    const totalLen = clips.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const c of clips) {
      merged.set(c, off);
      off += c.length;
    }
    audioRef.current = { samples: merged, sampleRate };
    setPhase("ready");
  }

  function resetIdle() {
    setPhase("idle");
    setTranscript("");
    setLines([]);
    setDone(0);
    setEta(null);
  }

  function cancel() {
    cancelRef.current = true;
    if (phase === "transcript") void cancelGeneration().catch(() => {});
    resetIdle();
  }

  function togglePlay() {
    if (playing) {
      playbackRef.current?.stop();
      playbackRef.current = null;
      setPlaying(false);
      return;
    }
    const a = audioRef.current;
    if (!a) return;
    const pb = playAudio(a.samples, a.sampleRate);
    playbackRef.current = pb;
    setPlaying(true);
    void pb.done.then(() => {
      playbackRef.current = null;
      setPlaying(false);
    });
  }

  async function exportAudio() {
    const a = audioRef.current;
    if (!a) return;
    try {
      const b64 = encodeAudio(a.samples);
      const stamp = new Date().toISOString().slice(0, 10);
      await exportWavFile(`deep-dive-${stamp}.wav`, b64, a.sampleRate);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const busy = phase === "transcript" || phase === "synth";
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const durationSec = audioRef.current
    ? audioRef.current.samples.length / audioRef.current.sampleRate
    : 0;

  return createPortal(
    <div className="preview-overlay" onMouseDown={busy ? undefined : onClose}>
      <div className="setup-modal podcast-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="setup-head">
          <div>
            <div className="setup-title"><IconMic size={18} /> {t("podcastTitle")}</div>
            <div className="setup-hw">{t("podcastSub")}</div>
          </div>
          {!busy && (
            <button className="preview-close" onClick={onClose}>
              <Icon name="x" size={12} strokeWidth={2.2} />
            </button>
          )}
        </div>

        {phase === "idle" && (
          <div className="kb-setup">
            <p className="kb-note">{t("podcastIntro")}</p>
            <button className="setup-dl ready" onClick={() => void run()}>
              {t("podcastStart")}
            </button>
          </div>
        )}

        {busy && (
          <div className="podcast-progress">
            <div className="podcast-phase">
              {phase === "transcript" ? t("podcastWriting") : t("podcastVoicing")}
            </div>
            {phase === "synth" && (
              <>
                <div className="setup-progress">
                  <div className="setup-progress-fill" style={{ width: `${pct}%` }} />
                  <span>
                    {done} / {total}
                  </span>
                </div>
                {eta !== null && (
                  <div className="podcast-eta">
                    {t("podcastEta")} ~{fmtTime(eta)}
                  </div>
                )}
              </>
            )}
            {phase === "transcript" && transcript && (
              <div className="podcast-stream">{transcript.slice(-600)}</div>
            )}
            <button className="setup-dl podcast-cancel" onClick={cancel}>
              {t("podcastCancel")}
            </button>
          </div>
        )}

        {phase === "ready" && (
          <div className="podcast-ready">
            <div className="podcast-meta">
              {lines.length} {t("podcastTurns")} · {fmtTime(durationSec)}
            </div>
            <div className="podcast-controls">
              <button className="setup-dl ready" onClick={togglePlay}>
                {playing ? (
                  <><IconStop size={13} style={{ marginRight: 6 }} /> {t("podcastStop")}</>
                ) : (
                  <><IconPlay size={13} style={{ marginRight: 6 }} /> {t("podcastPlay")}</>
                )}
              </button>
              <button className="setup-dl" onClick={() => void exportAudio()}>
                <IconDownload size={14} style={{ marginRight: 6 }} /> {t("podcastExport")}
              </button>
              <button className="setup-dl" onClick={() => void run()}>
                <IconRefresh size={14} style={{ marginRight: 6 }} /> {t("podcastRegen")}
              </button>
            </div>
            <div className="podcast-script">
              {lines.map((ln, i) => (
                <div key={i} className={`podcast-line ${ln.speaker === "A" ? "a" : "b"}`}>
                  <span className="podcast-speaker">{ln.speaker === "A" ? HOST_A : HOST_B}</span>
                  <span className="podcast-text">{ln.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {phase === "error" && (
          <div className="kb-setup">
            <div className="setup-err">{error.slice(0, 300)}</div>
            <button className="setup-dl" onClick={resetIdle}>
              {t("podcastRetry")}
            </button>
          </div>
        )}

        {lang === "zh" && <div className="setup-foot">{t("podcastFootZh")}</div>}
      </div>
    </div>,
    document.body,
  );
}
