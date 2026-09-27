import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { open as openDialog } from "@tauri-apps/plugin-dialog";

import { useI18n, type TKey } from "../lib/i18n";
import { watchContentHeight } from "../lib/autoGrow";
import {
  musicMediaUrl,
  openExternal,
  saveFileCopy,
  type ModelInfo,
  type MusicEdit,
  type MusicEditKind,
  type MusicFamilySpec,
  type MusicRecord,
} from "../lib/ipc";
import {
  IDEAS,
  SECTION_TAGS,
  buildMusicRequest,
  draftFromRecord,
  familyParams,
  fmtClock,
  isCustom,
  lengthOf,
  musicEta,
  musicPercent,
  paramValue,
  runStages,
  settingsFromRecord,
  shownLength,
  withParam,
  type MusicSettings,
} from "../lib/musicGen";
import { fmtDuration } from "../lib/imageGen";
import type { MusicRun, MusicStudioState } from "../lib/useMusicStudio";
import { Icon } from "./Icon";
import { MusicParamField, fill } from "./MusicParamField";
import { UserCopy, UserText } from "./UserText";
import { useConfirm } from "./ConfirmModal";

const STAGE_KEY: Record<string, TKey> = {
  prepare: "musStagePrepare",
  score: "musStageScore",
  tokens: "musStageTokens",
  render: "musStageRender",
  decode: "musStageDecode",
  save: "musStageSave",
};

const EDIT_KEY: Record<MusicEditKind, TKey> = {
  rearrange: "musEditRearrange",
  continue: "musEditContinue",
  repaint: "musEditRepaint",
  cover: "musEditCover",
  variation: "musEditVariation",
  inpaint: "musEditInpaint",
};

const EDIT_TIP: Record<MusicEditKind, TKey> = {
  rearrange: "musEditRearrangeTip",
  continue: "musEditContinueTip",
  repaint: "musEditRepaintTip",
  cover: "musEditCoverTip",
  variation: "musEditVariationTip",
  inpaint: "musEditInpaintTip",
};

const RANGE_EDITS: MusicEditKind[] = ["repaint", "inpaint"];

function basename(p: string): string {
  return p.split(/[/\\]/).pop() || p;
}
function dirname(p: string): string {
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return i > 0 ? p.slice(0, i) : p;
}

/** Click-outside-closing popover anchored above a composer control. */
function Pop({ open, onClose, children, className }: { open: boolean; onClose: () => void; children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.parentElement?.contains(e.target as Node)) onClose();
    };
    const esc = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", esc);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", esc);
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div ref={ref} className={`is-pop ${className ?? ""}`}>
      {children}
    </div>
  );
}

/** The one piece playing: starting another pauses it. */
let playingNow: HTMLAudioElement | null = null;

/** A waveform: the piece's outline, what has played, and a stretch selected
 *  by dragging (for the edits that replace one). */
function Waveform({
  peaks,
  duration,
  position,
  selection,
  selectable,
  onSeek,
  onSelect,
}: {
  peaks: number[];
  duration: number;
  position: number;
  selection: [number, number] | null;
  selectable: boolean;
  onSeek: (t: number) => void;
  onSelect: (sel: [number, number] | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; t: number; moved: boolean } | null>(null);
  const bars = peaks.length > 0 ? peaks : new Array(120).fill(0.12);
  const at = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r || r.width <= 0) return 0;
    return Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * duration;
  };
  const played = duration > 0 ? position / duration : 0;
  return (
    <div
      ref={ref}
      className={`ms-wave ${selectable ? "selectable" : ""}`}
      onPointerDown={(e) => {
        (e.target as Element).setPointerCapture?.(e.pointerId);
        drag.current = { x: e.clientX, t: at(e.clientX), moved: false };
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d || !selectable) return;
        if (Math.abs(e.clientX - d.x) > 4) d.moved = true;
        if (d.moved) {
          const t = at(e.clientX);
          onSelect([Math.min(d.t, t), Math.max(d.t, t)]);
        }
      }}
      onPointerUp={(e) => {
        const d = drag.current;
        drag.current = null;
        if (!d) return;
        if (!d.moved) {
          onSelect(null);
          onSeek(at(e.clientX));
        }
      }}
    >
      <svg viewBox={`0 0 ${bars.length * 3} 40`} preserveAspectRatio="none" aria-hidden="true">
        {bars.map((v, i) => {
          const h = Math.max(1.2, v * 38);
          return (
            <rect
              key={i}
              x={i * 3 + 0.4}
              y={20 - h / 2}
              width={2.2}
              height={h}
              rx={1}
              className={i / bars.length < played ? "on" : ""}
            />
          );
        })}
      </svg>
      {selection && duration > 0 && (
        <div
          className="ms-wave-sel"
          style={{ left: `${(selection[0] / duration) * 100}%`, width: `${((selection[1] - selection[0]) / duration) * 100}%` }}
        />
      )}
      {duration > 0 && <div className="ms-wave-head" style={{ left: `${played * 100}%` }} />}
    </div>
  );
}

/** A finished piece: the player, its outline, and what can be done with it. */
function Track({
  rec,
  spec,
  autoPlay,
  busy,
  onEdit,
  notify,
}: {
  rec: MusicRecord;
  spec: MusicFamilySpec | null;
  autoPlay: boolean;
  busy: boolean;
  onEdit: (edit: MusicEdit) => void;
  notify: (kind: "warn" | "error", text: string) => void;
}) {
  const { t } = useI18n();
  const audio = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState(0);
  const [dur, setDur] = useState(rec.audio.seconds || 0);
  const [sel, setSel] = useState<[number, number] | null>(null);
  const [failed, setFailed] = useState(false);
  const src = useMemo(() => musicMediaUrl(rec.audio.path), [rec.audio.path]);
  const edits = spec && spec.id === rec.family ? spec.edits : [];
  const canRange = edits.some((e) => RANGE_EDITS.includes(e));

  const play = useCallback(() => {
    const a = audio.current;
    if (!a) return;
    if (playingNow && playingNow !== a) playingNow.pause();
    playingNow = a;
    // A pause before it starts (AbortError) or a blocked autoplay
    // (NotAllowedError) leaves a good file paused; only a source that
    // cannot be played is one.
    void a.play().catch((e: unknown) => {
      if ((e as DOMException)?.name === "NotSupportedError") setFailed(true);
    });
  }, []);

  useEffect(() => {
    if (autoPlay) play();
  }, [autoPlay, play]);

  useEffect(() => {
    const a = audio.current;
    return () => {
      if (a && playingNow === a) {
        a.pause();
        playingNow = null;
      }
    };
  }, []);

  const toggle = () => {
    const a = audio.current;
    if (!a) return;
    if (a.paused) play();
    else a.pause();
  };

  const seek = (s: number) => {
    const a = audio.current;
    if (!a) return;
    a.currentTime = s;
    setPos(s);
  };

  const edit = (kind: MusicEditKind) => {
    const base: MusicEdit = { kind, parentId: rec.id, start: 0, end: 0, strength: 0 };
    if (kind === "continue") base.start = pos > 0.5 && pos < dur - 0.5 ? Math.round(pos) : 0;
    if (RANGE_EDITS.includes(kind)) {
      const [a, b] = sel ?? [Math.max(0, pos), Math.min(dur, pos + 10)];
      base.start = Math.round(a * 10) / 10;
      base.end = Math.round(b * 10) / 10;
    }
    onEdit(base);
  };

  return (
    <div className="ms-track">
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => setPos(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => Number.isFinite(e.currentTarget.duration) && setDur(e.currentTarget.duration)}
        onError={() => setFailed(true)}
      />
      <div className="ms-track-row">
        <button className={`ms-play ${playing ? "on" : ""}`} onClick={toggle} title={playing ? t("musPause") : t("musPlay")} disabled={failed}>
          {playing ? (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <rect x="6.5" y="5" width="4" height="14" rx="1.2" fill="currentColor" />
              <rect x="13.5" y="5" width="4" height="14" rx="1.2" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.2-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" fill="currentColor" />
            </svg>
          )}
        </button>
        <div className="ms-track-main">
          <Waveform peaks={rec.peaks} duration={dur} position={pos} selection={sel} selectable={canRange} onSeek={seek} onSelect={setSel} />
          <div className="ms-track-time">
            <span>{fmtClock(pos)}</span>
            {sel ? (
              <span className="ms-sel-label">
                {t("musSelected", { a: fmtClock(sel[0]), b: fmtClock(sel[1]) })}
              </span>
            ) : canRange ? (
              <span className="ms-sel-hint">{t("musSelectHint")}</span>
            ) : null}
            <span>{fmtClock(dur)}</span>
          </div>
        </div>
      </div>
      {failed && <div className="ms-track-err">{t("musPlayFailed")}</div>}
      <div className="ms-track-actions">
        {edits.map((k) => (
          <button key={k} className="ms-edit-btn" disabled={busy} title={t(EDIT_TIP[k])} onClick={() => edit(k)}>
            {t(EDIT_KEY[k])}
            {k === "continue" && pos > 0.5 && pos < dur - 0.5 ? ` · ${fmtClock(pos)}` : ""}
            {RANGE_EDITS.includes(k) && sel ? ` · ${fmtClock(sel[0])}–${fmtClock(sel[1])}` : ""}
          </button>
        ))}
        <span className="ms-track-tools">
          <button
            title={t("musActSave")}
            onClick={() => void saveFileCopy(rec.audio.path, basename(rec.audio.path)).catch((e) => notify("error", String(e)))}
          >
            <Icon name="download" size={14} />
          </button>
          {rec.audio.scorePath && (
            <button title={t("musActScore")} onClick={() => void openExternal(rec.audio.scorePath!).catch(console.error)}>
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <path d="M9 18V5l11-2v13" />
                <circle cx="6" cy="18" r="3" />
                <circle cx="17" cy="16" r="3" />
              </svg>
            </button>
          )}
          <button title={t("musActReveal")} onClick={() => void openExternal(dirname(rec.audio.path)).catch(console.error)}>
            <Icon name="folder" size={14} />
          </button>
        </span>
      </div>
    </div>
  );
}

/** The chips under a round: what it ran with. */
function ParamChips({ rec }: { rec: MusicRecord }) {
  const { t } = useI18n();
  const p = rec.params ?? {};
  const o = p.options ?? {};
  const chips = [
    fmtClock(rec.audio.seconds),
    o.cot === "off" ? t("musPlanOff") : o.cot === "melody" ? t("musPlanMelody") : null,
    o.num_inference_steps ? t("musStepsN", { n: o.num_inference_steps }) : null,
    `${t("musSeed")} ${rec.audio.seed}`,
    rec.audio.sampleRate ? `${(rec.audio.sampleRate / 1000).toFixed(rec.audio.sampleRate % 1000 ? 1 : 0)} kHz` : null,
    rec.elapsedMs ? fmtDuration(rec.elapsedMs) : null,
    rec.model || null,
  ].filter(Boolean) as string[];
  return (
    <div className="is-chips">
      {chips.map((c) => (
        <span key={c} className="is-chip">
          {c}
        </span>
      ))}
    </div>
  );
}

/** The piece being made: the overall percentage, the stages with the one
 *  under way, and what is known so far. */
function RunView({ run, spec, keptSeconds, onStop }: { run: MusicRun; spec: MusicFamilySpec; keptSeconds: number; onStop: () => void }) {
  const { t } = useI18n();
  const [now, setNow] = useState(Date.now());
  const pctRef = useRef(0);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(id);
  }, []);
  const pct = musicPercent(spec, run.request, run.progress, pctRef.current, keptSeconds);
  pctRef.current = pct;
  const stages = runStages(spec, run.request).map(([s]) => s);
  const cur = run.progress.stage;
  const curIdx = stages.indexOf(cur);
  const elapsed = now - run.startedAt;
  const eta = musicEta(pct, elapsed);
  const p = run.progress;
  const detail: string[] = [];
  if (cur === "tokens" && spec.tokenRate > 0 && p.done > 0) detail.push(t("musComposedSoFar", { t: fmtClock(p.done / spec.tokenRate) }));
  else if (cur === "score" && p.done > 0) detail.push(t("musScoreTokens", { n: p.done }));
  else if (p.total > 0 && cur !== "tokens") detail.push(`${Math.round((p.done / p.total) * 100)}%`);
  else if (cur === "tokens" && p.total > 0) detail.push(`${p.done}/${p.total}`);
  if (p.seconds != null) detail.push(t("musComposed", { t: fmtClock(p.seconds) }));
  detail.push(t("musElapsed", { t: fmtDuration(elapsed) }));
  if (eta != null) detail.push(t("musEta", { t: fmtDuration(eta * 1000) }));

  return (
    <div className="ms-run">
      <div className="ms-run-top">
        <div className="ms-run-pct">
          <span className="ms-run-num">{Math.round(pct)}</span>
          <span className="ms-run-sign">%</span>
        </div>
      </div>
      <ol className="ms-stages">
        {stages.map((s, i) => {
          const state = cur === "save" || i < curIdx ? "done" : i === curIdx ? "active" : "todo";
          return (
            <li key={s} className={state}>
              <span className="ms-stage-dot">{state === "done" ? <Icon name="check" size={10} strokeWidth={2.6} /> : i + 1}</span>
              <span className="ms-stage-name">{t(STAGE_KEY[s] ?? "musStagePrepare")}</span>
            </li>
          );
        })}
      </ol>
      <div className="is-run-bar" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <div className="is-run-fill" style={{ width: `${Math.max(1.5, pct)}%` }} />
      </div>
      <div className="is-run-info">
        <span className="is-run-label">
          <span className="cm-spin" /> {run.stopping ? t("musStopping") : t(STAGE_KEY[cur] ?? "musStagePrepare")}
        </span>
        <span className="is-run-detail">{detail.join(" · ")}</span>
      </div>
      <div className="is-run-actions">
        <button className="is-btn danger" disabled={run.stopping} onClick={onStop}>
          {t("musStop")}
        </button>
      </div>
    </div>
  );
}

/** A round's request, as the user's side of the thread. */
function RequestBubble({
  rec,
  prompt,
  lyrics,
  instrumental,
  edit,
  fromRound,
  onJump,
  onReuse,
}: {
  rec?: MusicRecord;
  prompt: string;
  lyrics: string;
  instrumental: boolean;
  edit: MusicEdit | null | undefined;
  fromRound: number | null;
  onJump?: () => void;
  onReuse?: () => void;
}) {
  const { t } = useI18n();
  const copy = [prompt, lyrics].filter(Boolean).join("\n\n");
  void rec;
  return (
    <div className="msg user">
      <div className="bubble ms-bubble" data-copy={copy}>
        {edit && fromRound != null && (
          <button className="is-from" onClick={onJump} title={t("musJumpToRound")}>
            <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M9 14l-4-4 4-4M5 10h9a5 5 0 0 1 5 5v4" />
            </svg>
            {t(EDIT_KEY[edit.kind])} · {t("musFromRound", { n: fromRound })}
            {RANGE_EDITS.includes(edit.kind) ? ` · ${fmtClock(edit.start)}–${edit.end > 0 ? fmtClock(edit.end) : "∞"}` : ""}
            {edit.kind === "continue" && edit.start > 0 ? ` · ${t("musKeepFirst", { t: fmtClock(edit.start) })}` : ""}
          </button>
        )}
        {prompt && (
          <span className="ms-bubble-style">
            <Icon name="music" size={12} strokeWidth={1.9} />
            <UserText content={prompt} expandLabel={t("expandAll")} collapseLabel={t("collapseText")} />
          </span>
        )}
        {instrumental ? (
          <span className="ms-bubble-inst">{t("musModeInstrumental")}</span>
        ) : lyrics ? (
          <div className="ms-bubble-lyrics">
            <UserText content={lyrics} expandLabel={t("expandAll")} collapseLabel={t("collapseText")} />
          </div>
        ) : null}
        <UserCopy content={copy} title={t("copyMsg")} />
        {onReuse && (
          <button className="user-edit" title={t("musReuse")} onClick={onReuse}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <path d="M14.5 5.5l4 4M4 20l1-4L16 5a2 2 0 0 1 3 3L8 19l-4 1z" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

export function MusicStudio({
  model,
  studio,
  settings,
  onSettings,
  onOpenSettings,
  notify,
  sendKey,
}: {
  model: ModelInfo;
  studio: MusicStudioState;
  settings: MusicSettings;
  onSettings: (patch: Partial<MusicSettings>) => void;
  /** "More settings" — the music tab of the settings panel. */
  onOpenSettings: () => void;
  notify: (kind: "warn" | "error", text: string) => void;
  sendKey: "enter" | "modEnter";
}) {
  const { t, lang } = useI18n();
  const confirm = useConfirm();
  const info = model.music!;
  const spec = info.spec;
  const [pop, setPop] = useState<"" | "length" | "params" | "plan">("");
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const lyricsRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const run = studio.run;
  const busy = !!run;
  const shownRun = studio.runHere ? run : null;
  const rounds = studio.rounds;
  const d = studio.draft;
  const roundIndex = useMemo(() => new Map(rounds.map((r, i) => [r.id, i])), [rounds]);
  const params = useMemo(() => familyParams(spec.id, info.requestOptions), [spec.id, info.requestOptions]);
  const mainParams = params.filter((p) => p.group === "main");
  const sings = !!spec.lyrics;
  const song = sings && d.mode === "song";
  const len = shownLength(settings, spec);

  useLayoutEffect(() => {
    const el = promptRef.current;
    const row = el?.parentElement;
    if (!el || !row) return;
    return watchContentHeight(el, row, 120);
  }, [d.prompt]);
  // The lyrics grow with what is written, up to about a verse and a chorus.
  useLayoutEffect(() => {
    const el = lyricsRef.current;
    if (!el) return;
    return watchContentHeight(el, el.parentElement ?? el, Math.round(window.innerHeight * 0.26));
  }, [d.lyrics, song]);

  // An empty studio (the greeting and ideas) rests at its top: in a short
  // window following the end would hide the greeting under the composer.
  const emptyRef = useRef(false);
  emptyRef.current = rounds.length === 0 && !shownRun;
  const toBottom = useCallback((smooth: boolean) => {
    const el = scrollRef.current;
    if (!el) return;
    const top = emptyRef.current ? 0 : el.scrollHeight;
    el.scrollTo({ top, behavior: smooth ? "smooth" : ("instant" as ScrollBehavior) });
  }, []);

  useLayoutEffect(() => {
    followRef.current = true;
    setShowJump(false);
    toBottom(false);
  }, [studio.sessionId, toBottom]);

  useEffect(() => {
    followRef.current = true;
    toBottom(true);
  }, [shownRun?.id, rounds.length, toBottom]);

  useEffect(() => {
    const inner = innerRef.current;
    if (!inner || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (followRef.current) toBottom(false);
    });
    ro.observe(inner);
    return () => ro.disconnect();
  }, [toBottom]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    followRef.current = atEnd;
    setShowJump(!atEnd);
  };

  const jumpTo = (id: string) => {
    followRef.current = false;
    document.getElementById(`ms-round-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const parentOf = (id: string | null | undefined) => (id ? rounds.find((r) => r.id === id) ?? null : null);
  const fromRound = (id: string | null | undefined): number | null => {
    const i = id ? roundIndex.get(id) : undefined;
    return i === undefined ? null : i + 1;
  };

  // What the composer lacks before it can send.
  const missing: string | null = !d.prompt.trim() && (spec.promptOption === "style" || !spec.promptText)
    ? t("musNeedStyle")
    : !d.prompt.trim() && !d.lyrics.trim()
      ? t("musNeedSomething")
      : song && spec.lyricsRequired && !d.lyrics.trim()
        ? t("musNeedLyrics")
        : null;

  const go = () => {
    if (busy) return;
    if (missing) {
      notify("warn", missing);
      return;
    }
    void studio.generate(buildMusicRequest(d, settings, spec));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== "Enter" || e.nativeEvent.isComposing) return;
    // Lyrics are lines: Enter is a new line there, and only the send
    // shortcut sends.
    const inLyrics = e.currentTarget === lyricsRef.current;
    if (sendKey === "modEnter" || inLyrics) {
      if (e.metaKey || e.ctrlKey) {
        e.preventDefault();
        go();
      }
    } else if (!e.shiftKey) {
      e.preventDefault();
      go();
    }
  };

  const insertTag = (tag: string) => {
    const el = lyricsRef.current;
    const text = d.lyrics;
    const at = el ? el.selectionStart : text.length;
    const before = text.slice(0, at);
    const after = text.slice(at);
    const piece = `${before && !before.endsWith("\n") ? "\n" : ""}${tag}\n`;
    studio.patchDraft({ lyrics: before + piece + after });
    requestAnimationFrame(() => {
      if (!el) return;
      el.focus();
      const p = before.length + piece.length;
      el.setSelectionRange(p, p);
    });
  };

  const pickScore = async () => {
    const picked = await openDialog({ multiple: false, filters: [{ name: "ABC", extensions: ["abc", "txt"] }] });
    if (typeof picked === "string") studio.patchDraft({ scorePath: picked });
  };

  const reuse = (rec: MusicRecord) => {
    studio.setDraft(draftFromRecord(rec));
    if (rec.family === spec.id) onSettings(settingsFromRecord(rec, settings));
    promptRef.current?.focus();
  };

  const again = (rec: MusicRecord) => {
    if (busy) return;
    const p = rec.params ?? {};
    void studio.generate({
      prompt: rec.prompt,
      lyrics: rec.lyrics,
      instrumental: !!p.instrumental,
      seconds: p.seconds ?? 0,
      seed: -1,
      options: p.options ?? {},
      scorePath: p.scorePath ?? null,
      edit: p.edit ?? null,
      outDir: settings.musOutputDir.trim() || null,
    });
  };

  const del = async (rec: MusicRecord) => {
    const ok = await confirm({ title: t("musDeleteTitle"), message: t("musDeleteConfirm"), confirmLabel: t("confirmDelete"), danger: true });
    if (ok) await studio.removeRound(rec.id).catch((e) => notify("error", String(e)));
  };

  const startEdit = (edit: MusicEdit) => {
    studio.setEdit(edit);
    // A rearrangement or continuation starts from the same words, to change.
    const parent = parentOf(edit.parentId);
    if (parent && !d.prompt.trim() && !d.lyrics.trim()) studio.patchDraft({ prompt: parent.prompt, lyrics: parent.lyrics, edit });
    promptRef.current?.focus();
  };

  const editParent = parentOf(d.edit?.parentId);
  const editFrom = fromRound(d.edit?.parentId);
  const ideas = lang === "zh" ? IDEAS.zh : IDEAS.en;
  // What this family does: sing (and whether it can leave the words out),
  // and the edits it makes of a piece.
  const heroBase = t(!sings ? "musHeroSubMusic" : spec.instrumentalLyrics == null ? "musHeroSubLyrics" : "musHeroSubSong", { model: info.familyName });
  const editNames = spec.edits.map((k) => (lang === "zh" ? t(EDIT_KEY[k]) : t(EDIT_KEY[k]).toLowerCase()));
  const heroSub = editNames.length
    ? t("musHeroSubEdits", { base: heroBase, edits: editNames.join(lang === "zh" ? "、" : lang === "en" ? " or " : ", ") })
    : heroBase;
  const keptSeconds =
    shownRun?.request.edit?.kind === "continue"
      ? shownRun.request.edit.start || parentOf(shownRun.request.edit.parentId)?.audio.seconds || 0
      : 0;
  const customCount = Object.keys(settings.musParams[spec.id] ?? {}).length;

  return (
    <>
      <div className="chat-wrap">
        <main className="chat is-thread ms-thread" ref={scrollRef} onScroll={onScroll}>
          <div ref={innerRef}>
            {rounds.length === 0 && !shownRun ? (
              <div className="empty is-empty ms-empty">
                <div className="empty-hero">
                  <div className="ms-hero-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                      <path d="M9 18V5l11-2v13" />
                      <circle cx="6" cy="18" r="3" />
                      <circle cx="17" cy="16" r="3" />
                    </svg>
                  </div>
                  <div className="empty-greeting">{t("musHero")}</div>
                  <div className="empty-sub">{heroSub}</div>
                </div>
                <div className="suggestions">
                  {ideas
                    .filter((i) => sings || !i.lyrics)
                    .map((s) => (
                      <button
                        key={s.prompt}
                        className="suggestion"
                        onClick={() => {
                          studio.patchDraft({ prompt: s.prompt, lyrics: s.lyrics, mode: s.lyrics || !spec.instrumentalLyrics ? "song" : "instrumental" });
                          promptRef.current?.focus();
                        }}
                      >
                        {s.prompt}
                        {s.lyrics && <small className="ms-idea-lyrics">{s.lyrics.split("\n").filter((l) => !l.startsWith("[")).slice(0, 2).join(" / ")}</small>}
                      </button>
                    ))}
                </div>
              </div>
            ) : (
              <>
                {rounds.map((rec) => {
                  const p = rec.params ?? {};
                  return (
                    <div key={rec.id} className="is-round" id={`ms-round-${rec.id}`}>
                      <RequestBubble
                        rec={rec}
                        prompt={rec.prompt}
                        lyrics={rec.lyrics}
                        instrumental={!!p.instrumental}
                        edit={p.edit}
                        fromRound={fromRound(rec.parentId)}
                        onJump={rec.parentId ? () => jumpTo(rec.parentId!) : undefined}
                        onReuse={busy ? undefined : () => reuse(rec)}
                      />
                      <div className="msg assistant is-reply">
                        <Track
                          rec={rec}
                          spec={spec}
                          autoPlay={studio.fresh === rec.id && settings.musAutoPlay}
                          busy={busy}
                          onEdit={startEdit}
                          notify={notify}
                        />
                        <ParamChips rec={rec} />
                        <div className="msg-actions">
                          <button className="msg-action" title={t("musAgainTitle")} onClick={() => again(rec)} disabled={busy || rec.family !== spec.id}>
                            {t("musAgain")}
                          </button>
                          <button className="msg-action" onClick={() => reuse(rec)}>
                            {t("musReuse")}
                          </button>
                          <button className="msg-action" onClick={() => void del(rec)} disabled={busy}>
                            {t("musDelete")}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
                {shownRun && (
                  <div className="is-round">
                    <RequestBubble
                      prompt={shownRun.request.prompt}
                      lyrics={shownRun.request.lyrics}
                      instrumental={shownRun.request.instrumental}
                      edit={shownRun.request.edit}
                      fromRound={fromRound(shownRun.request.edit?.parentId)}
                      onJump={shownRun.request.edit ? () => jumpTo(shownRun.request.edit!.parentId) : undefined}
                    />
                    <div className="msg assistant is-reply">
                      <RunView run={shownRun} spec={spec} keptSeconds={keptSeconds} onStop={() => void studio.cancel()} />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </main>
        {showJump && (rounds.length > 0 || shownRun) && (
          <button
            className="jump-bottom"
            title={t("jumpLatest")}
            onClick={() => {
              followRef.current = true;
              toBottom(true);
            }}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>

      <footer className="composer is-composer ms-composer">
        {(d.edit || (spec.planning && d.scorePath)) && (
          <div className="is-extras">
            {d.edit && (
              <div className="ms-edit-chip">
                <span className="ms-edit-title">
                  {t(EDIT_KEY[d.edit.kind])}
                  {editFrom != null && <em className="is-ref-from">{t("musFromRound", { n: editFrom })}</em>}
                  {editParent ? <small> · {fmtClock(editParent.audio.seconds)}</small> : null}
                </span>
                {d.edit.kind === "continue" && (
                  <label className="ms-edit-field" title={t("musKeepHint")}>
                    {t("musKeep")}
                    <input
                      type="number"
                      min={0}
                      max={editParent?.audio.seconds ?? 600}
                      step={1}
                      value={d.edit.start}
                      onChange={(e) => studio.setEdit({ ...d.edit!, start: Math.max(0, Number(e.target.value) || 0) })}
                    />
                    <span>{d.edit.start > 0 ? t("musSeconds") : t("musKeepAll")}</span>
                  </label>
                )}
                {RANGE_EDITS.includes(d.edit.kind) && (
                  <label className="ms-edit-field">
                    {t("musFrom")}
                    <input type="number" min={0} step={0.5} value={d.edit.start} onChange={(e) => studio.setEdit({ ...d.edit!, start: Math.max(0, Number(e.target.value) || 0) })} />
                    {t("musTo")}
                    <input type="number" min={0} step={0.5} value={d.edit.end} onChange={(e) => studio.setEdit({ ...d.edit!, end: Math.max(0, Number(e.target.value) || 0) })} />
                    <span>{t("musSeconds")}</span>
                  </label>
                )}
                {d.edit.kind === "variation" && (
                  <label className="ms-edit-field" title={t("musStrengthHint")}>
                    {t("musStrength")} <b>{(d.edit.strength || 0.7).toFixed(2)}</b>
                    <input
                      type="range"
                      min={0.05}
                      max={1}
                      step={0.05}
                      value={d.edit.strength || 0.7}
                      style={fill(d.edit.strength || 0.7, 0.05, 1)}
                      onChange={(e) => studio.setEdit({ ...d.edit!, strength: Number(e.target.value) })}
                    />
                  </label>
                )}
                <button className="attach-remove" title={t("musEditRemove")} onClick={() => studio.setEdit(null)}>
                  <Icon name="x" size={11} strokeWidth={2.2} />
                </button>
              </div>
            )}
            {spec.planning && d.scorePath && (
              <div className="ms-edit-chip">
                <span className="ms-edit-title">
                  {t("musScoreFollow")} <small>· {basename(d.scorePath)}</small>
                </span>
                <button className="attach-remove" title={t("musScoreRemove")} onClick={() => studio.patchDraft({ scorePath: null })}>
                  <Icon name="x" size={11} strokeWidth={2.2} />
                </button>
              </div>
            )}
          </div>
        )}

        {sings && (
          <div className="ms-modes" role="tablist">
            <button className={d.mode === "song" ? "on" : ""} onClick={() => studio.patchDraft({ mode: "song" })}>
              <Icon name="mic" size={12} strokeWidth={1.9} /> {t("musModeSong")}
            </button>
            {spec.instrumentalLyrics != null && (
              <button className={d.mode === "instrumental" ? "on" : ""} onClick={() => studio.patchDraft({ mode: "instrumental" })}>
                <Icon name="music" size={12} strokeWidth={1.9} /> {t("musModeInstrumental")}
              </button>
            )}
            {spec.planning && (
              <button className={d.scorePath ? "on" : ""} title={t("musScoreHint")} onClick={() => void pickScore()}>
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" aria-hidden="true">
                  <path d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                </svg>
                {t("musModeScore")}
              </button>
            )}
          </div>
        )}

        {/* One card: the style on its first line, the lyrics under a hairline
            when the piece is sung, the button at its foot. */}
        <div className={`input-row is-input ms-input ${song ? "with-lyrics" : ""}`}>
          <div className="ms-fields">
            <textarea
              ref={promptRef}
              value={d.prompt}
              onChange={(e) => studio.patchDraft({ prompt: e.target.value })}
              onKeyDown={onKeyDown}
              placeholder={spec.promptTemplate ? t("musPromptPhTagged") : sings ? t("musStylePh") : t("musPromptPh")}
              rows={1}
            />
            {song && (
              <div className="ms-lyrics">
                <textarea
                  ref={lyricsRef}
                  value={d.lyrics}
                  onChange={(e) => studio.patchDraft({ lyrics: e.target.value })}
                  onKeyDown={onKeyDown}
                  placeholder={spec.lyricsRequired ? t("musLyricsPhRequired") : t("musLyricsPh")}
                  rows={4}
                  spellCheck={false}
                />
                <div className="ms-tags">
                  {SECTION_TAGS.map((tag) => (
                    <button key={tag} onClick={() => insertTag(tag)} title={t("musTagInsert")}>
                      {tag}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
          {busy ? (
            <button className="send-btn stop" onClick={() => void studio.cancel()} title={t("musStop")}>
              <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="3" fill="currentColor" />
              </svg>
            </button>
          ) : (
            <button className="send-btn is-go" onClick={go} disabled={!!missing} title={missing ?? t("musGenerate")}>
              <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <path d="M9 18V5l11-2v13" />
                <circle cx="6" cy="18" r="3" fill="currentColor" />
                <circle cx="17" cy="16" r="3" fill="currentColor" />
              </svg>
            </button>
          )}
        </div>

        <div className="is-bar">
          {spec.planning && (
            <div className="is-bar-item">
              <button className={`is-bar-btn ${pop === "plan" ? "on" : ""}`} title={t("musPlanTip")} onClick={() => setPop(pop === "plan" ? "" : "plan")}>
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                  <path d="M4 7h16M4 12h10M4 17h6" />
                </svg>
                {settings.musPlanning === "off" ? t("musPlanOff") : settings.musPlanning === "melody" ? t("musPlanMelody") : t("musPlanFull")}
              </button>
              <Pop open={pop === "plan"} onClose={() => setPop("")} className="ms-pop-plan">
                <div className="is-pop-title">{t("musPlanning")}</div>
                {(["full", "melody", "off"] as const).map((m) => (
                  <button key={m} className={`ms-pop-opt ${settings.musPlanning === m ? "on" : ""}`} onClick={() => onSettings({ musPlanning: m })}>
                    <b>
                      {m === "full" ? t("musPlanFull") : m === "melody" ? t("musPlanMelody") : t("musPlanOff")}
                      {m === "full" && <em> · {t("musRecommended")}</em>}
                    </b>
                    <small>{m === "full" ? t("musPlanFullTip") : m === "melody" ? t("musPlanMelodyTip") : t("musPlanOffTip")}</small>
                  </button>
                ))}
              </Pop>
            </div>
          )}

          {spec.length && (
            <div className="is-bar-item">
              <button className={`is-bar-btn ${pop === "length" ? "on" : ""}`} onClick={() => setPop(pop === "length" ? "" : "length")}>
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                  <circle cx="12" cy="13" r="8" />
                  <path d="M12 9v4l2.5 2M9 2h6" />
                </svg>
                {len > 0 ? `${spec.length.isLimit ? "≤ " : ""}${fmtClock(len)}` : t("musLengthAuto")}
              </button>
              <Pop open={pop === "length"} onClose={() => setPop("")} className="ms-pop-length">
                <div className="is-pop-title">{spec.length.isLimit ? t("musLengthLimit") : t("musLength")}</div>
                <div className="is-bases">
                  <button className={lengthOf(settings, spec) <= 0 ? "on" : ""} onClick={() => onSettings({ musLength: { ...settings.musLength, [spec.id]: 0 } })}>
                    {spec.length.defaultS > 0 ? `${t("musRecommended")} · ${fmtClock(spec.length.defaultS)}` : t("musLengthAuto")}
                  </button>
                  {[30, 60, 120, 180, 240]
                    .filter((s) => s >= spec.length!.minS && s <= spec.length!.maxS && s !== spec.length!.defaultS)
                    .map((s) => (
                      <button key={s} className={lengthOf(settings, spec) === s ? "on" : ""} onClick={() => onSettings({ musLength: { ...settings.musLength, [spec.id]: s } })}>
                        {fmtClock(s)}
                      </button>
                    ))}
                </div>
                <label className="is-field">
                  <span>
                    {t("musLengthCustom")} <b>{fmtClock(len || spec.length.minS)}</b>
                  </span>
                  <input
                    type="range"
                    min={spec.length.minS}
                    max={spec.length.maxS}
                    step={5}
                    value={len || spec.length.minS}
                    style={fill(len || spec.length.minS, spec.length.minS, spec.length.maxS)}
                    onChange={(e) => onSettings({ musLength: { ...settings.musLength, [spec.id]: Number(e.target.value) } })}
                  />
                </label>
                {spec.length.isLimit && <small className="ms-pop-note">{t("musLengthLimitTip")}</small>}
              </Pop>
            </div>
          )}

          <div className="is-bar-item">
            <button
              className={`is-bar-btn ${settings.musSeedLock ? "on" : ""}`}
              title={settings.musSeedLock ? t("musSeedLocked") : t("musSeedRandomTip")}
              onClick={() => onSettings({ musSeedLock: !settings.musSeedLock })}
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <rect x="4" y="4" width="16" height="16" rx="3" />
                <circle cx="9" cy="9" r="1" fill="currentColor" />
                <circle cx="15" cy="15" r="1" fill="currentColor" />
                {!settings.musSeedLock && <circle cx="15" cy="9" r="1" fill="currentColor" />}
                {!settings.musSeedLock && <circle cx="9" cy="15" r="1" fill="currentColor" />}
              </svg>
              {settings.musSeedLock ? `${t("musSeed")} ${settings.musSeed}` : t("musSeedRandom")}
            </button>
          </div>

          <div className="is-bar-item">
            <button className={`is-bar-btn ${pop === "params" ? "on" : ""}`} onClick={() => setPop(pop === "params" ? "" : "params")}>
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
                <circle cx="16" cy="7" r="2" />
                <circle cx="10" cy="17" r="2" />
              </svg>
              {customCount > 0 ? t("musParamsCustom", { n: customCount }) : t("musParamsRecommended")}
            </button>
            <Pop open={pop === "params"} onClose={() => setPop("")} className="is-pop-params ms-pop-params">
              {mainParams.map((p) => (
                <MusicParamField
                  key={p.key}
                  p={p}
                  compact
                  value={paramValue(settings, spec.id, p)}
                  custom={isCustom(settings, spec.id, p.key)}
                  onChange={(v) => onSettings(withParam(settings, spec.id, p.key, v))}
                />
              ))}
              {settings.musSeedLock && (
                <label className="is-field">
                  <span>{t("musSeed")}</span>
                  <span className="is-seed-row">
                    <input type="number" min={0} value={settings.musSeed} onChange={(e) => onSettings({ musSeed: Math.max(0, Number(e.target.value) || 0) })} />
                    <button className="is-btn" title={t("musSeedDice")} onClick={() => onSettings({ musSeed: Math.floor(Math.random() * 2147483647) })}>
                      <Icon name="refresh" size={13} strokeWidth={1.9} />
                    </button>
                  </span>
                </label>
              )}
              <div className="is-pop-foot">
                <button
                  className="is-btn"
                  onClick={() =>
                    onSettings({
                      musParams: { ...settings.musParams, [spec.id]: {} },
                      musLength: { ...settings.musLength, [spec.id]: 0 },
                      musPlanning: "full",
                    })
                  }
                >
                  {t("musResetRecommended")}
                </button>
                <button
                  className="is-btn"
                  onClick={() => {
                    setPop("");
                    onOpenSettings();
                  }}
                >
                  {t("musMoreSettings")}
                </button>
              </div>
            </Pop>
          </div>
        </div>
      </footer>
    </>
  );
}
