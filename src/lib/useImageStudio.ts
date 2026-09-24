// The image studio's state, shaped like the chat's: a list of sessions, the
// one open with its rounds, and the round being drawn. Shared by the sidebar
// (sessions) and the studio (thread and composer), so it lives above both —
// in App — as one hook. It outlives a model switch: loading another image
// model carries on in the session that is open.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  imageAttach,
  imageCancel,
  imageGenerate,
  imageGenerationDelete,
  imageSessionDelete,
  imageSessionDraft,
  imageSessionGet,
  imageSessionList,
  imageSessionRename,
  imageSessionSave,
  imageSessionSetPinned,
  logAppError,
  type ImageEvent,
  type ImageItem,
  type ImageRecord,
  type ImageRequest,
  type ImageSession,
  type SdEvent,
} from "./ipc";
import { convTitle } from "./fmt";
import { percent, referenceOf, smooth, type GenState } from "./imageGen";

/** A round being drawn, as the studio shows it. */
export interface ImageRun {
  id: string;
  request: ImageRequest;
  startedAt: number;
  gen: GenState;
  /** Overall percentage, never moving backwards. */
  pct: number;
  /** The seed of the picture being drawn. */
  seed: number | null;
  /** Latest denoised preview (a small JPEG data URL). */
  preview: string | null;
  /** Pictures already finished (a batch delivers them one by one). */
  images: ImageItem[];
  /** A stop was asked for: "all" now, "after" once this picture is done. */
  stopping: "" | "all" | "after";
  /** Work the engine's caches saved: the prompt's encoding reused, and
   *  denoising steps skipped so far. */
  cache: { encode: boolean; skipped: number; total: number };
}

/** A picture to start from (img2img) or to edit (editing models). */
export interface ImageRef {
  path: string;
  /** The round it came from, when it is one of this session's pictures. */
  parentId?: string | null;
}

/** What the composer holds, kept per session like an unsent message. */
export interface ImageDraft {
  prompt: string;
  /** null = the default negative prompt from Settings. */
  negative: string | null;
  reference: ImageRef | null;
}

const EMPTY_DRAFT: ImageDraft = { prompt: "", negative: null, reference: null };
/** The session open last, reopened on the next start. */
const LAST_SESSION_KEY = "chaty.imageSession";
/** The draft of a session not yet started (no round sent). */
const NEW_DRAFT_KEY = "chaty.imageDraft";

const FRESH: GenState = { stage: "encode", index: 0, count: 1, step: 0, steps: 0, secsPerStep: 0 };
const uid = () => Math.random().toString(36).slice(2);

function readLocal(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}
function writeLocal(key: string, value: string) {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* private window */
  }
}

/** A stored draft, whatever state it was left in. */
export function parseDraft(raw: string): ImageDraft {
  try {
    const d = JSON.parse(raw) as Partial<ImageDraft>;
    return {
      prompt: typeof d.prompt === "string" ? d.prompt : "",
      negative: typeof d.negative === "string" ? d.negative : null,
      reference: d.reference && typeof d.reference.path === "string" ? { path: d.reference.path, parentId: d.reference.parentId ?? null } : null,
    };
  } catch {
    return EMPTY_DRAFT;
  }
}

function draftJson(d: ImageDraft): string {
  return d.prompt || d.negative !== null || d.reference ? JSON.stringify(d) : "";
}

function apply(run: ImageRun, ev: SdEvent): ImageRun {
  switch (ev.type) {
    case "stage": {
      const gen: GenState = {
        ...run.gen,
        stage: ev.stage,
        index: ev.index,
        count: Math.max(1, ev.count),
        // A new phase starts its own count; the step rate carries over.
        step: 0,
        steps: ev.stage === run.gen.stage ? run.gen.steps : 0,
      };
      return { ...run, gen, pct: percent(gen, run.pct), seed: ev.seed ?? run.seed };
    }
    case "progress": {
      const gen: GenState = {
        ...run.gen,
        stage: ev.stage || run.gen.stage,
        step: ev.step,
        steps: ev.steps,
        secsPerStep: ev.stage === "sample" && ev.step > 0 ? smooth(run.gen.secsPerStep, ev.secs) : run.gen.secsPerStep,
      };
      return { ...run, gen, pct: percent(gen, run.pct) };
    }
    case "preview":
      return { ...run, preview: ev.dataUrl };
    case "image":
      return {
        ...run,
        images: [...run.images, { path: ev.path, width: ev.width, height: ev.height, seed: ev.seed }],
      };
    case "cache":
      return ev.kind === "conditioning"
        ? { ...run, cache: { ...run.cache, encode: true } }
        : { ...run, cache: { ...run.cache, skipped: run.cache.skipped + ev.skipped, total: run.cache.total + ev.total } };
  }
  return run;
}

/** A round that made nothing (stopped at once, or failed) gives its prompt
 *  back to an empty composer, to adjust and send again. */
export function restored(d: ImageDraft, request: ImageRequest): ImageDraft {
  if (d.prompt.trim()) return d;
  const ref = referenceOf(request);
  return {
    ...d,
    prompt: request.prompt,
    reference: d.reference ?? (ref ? { path: ref, parentId: request.parentId ?? null } : null),
  };
}

function freshRun(id: string, request: ImageRequest, startedAt: number): ImageRun {
  return {
    id,
    request,
    startedAt,
    gen: { ...FRESH, count: Math.max(1, request.batchCount) },
    pct: 0,
    seed: null,
    preview: null,
    images: [],
    stopping: "",
    cache: { encode: false, skipped: 0, total: 0 },
  };
}

export function useImageStudio({
  active,
  onBusy,
  notify,
  stoppedText,
  autoChain,
}: {
  /** An image model is loaded — the studio is on screen. */
  active: boolean;
  onBusy: (busy: boolean) => void;
  notify: (kind: "warn" | "error", text: string) => void;
  /** Toast for a generation stopped before it finished a picture. */
  stoppedText: string;
  /** Carry each round's picture into the next as the one to edit (an
   *  editing model, with the setting on). */
  autoChain: boolean;
}) {
  const [sessions, setSessions] = useState<ImageSession[]>([]);
  /** The open session; null = a new one, not started yet. */
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [rounds, setRounds] = useState<ImageRecord[]>([]);
  const [run, setRun] = useState<ImageRun | null>(null);
  const [draft, setDraft] = useState<ImageDraft>(EMPTY_DRAFT);

  const runRef = useRef<ImageRun | null>(null);
  runRef.current = run;
  const sessionRef = useRef<string | null>(null);
  sessionRef.current = sessionId;
  const busyRef = useRef(onBusy);
  busyRef.current = onBusy;
  const notifyRef = useRef(notify);
  notifyRef.current = notify;
  const chainRef = useRef(autoChain);
  chainRef.current = autoChain;
  /** A draft typed but not yet written, and where it belongs. */
  const pendingDraft = useRef<{ session: string | null; json: string } | null>(null);
  const started = useRef(false);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await imageSessionList());
    } catch (e) {
      console.error(e);
    }
  }, []);

  const writeDraft = useCallback((session: string | null, json: string) => {
    if (session) void imageSessionDraft(session, json).catch(() => {});
    else writeLocal(NEW_DRAFT_KEY, json);
  }, []);

  /** Write the draft being typed now, before the session changes under it. */
  const flushDraft = useCallback(() => {
    const p = pendingDraft.current;
    pendingDraft.current = null;
    if (p) writeDraft(p.session, p.json);
  }, [writeDraft]);

  // The composer is saved as it is typed (debounced), per session.
  useEffect(() => {
    if (!started.current) return;
    const json = draftJson(draft);
    pendingDraft.current = { session: sessionId, json };
    const id = window.setTimeout(flushDraft, 400);
    return () => window.clearTimeout(id);
  }, [draft, sessionId, flushDraft]);

  const show = useCallback((id: string | null, recs: ImageRecord[], d: ImageDraft) => {
    setSessionId(id);
    setRounds(recs);
    setDraft(d);
    writeLocal(LAST_SESSION_KEY, id ?? "");
  }, []);

  /** Open a session: its rounds and the prompt it was left with. */
  const openSession = useCallback(
    async (id: string) => {
      if (runRef.current || id === sessionRef.current) return;
      flushDraft();
      try {
        const data = await imageSessionGet(id);
        if (!data) {
          void refreshSessions();
          return;
        }
        show(id, data.records, parseDraft(data.draft));
      } catch (e) {
        console.error(e);
      }
    },
    [flushDraft, refreshSessions, show],
  );

  /** A new session: nothing on the canvas, an empty prompt. */
  const startNew = useCallback(() => {
    if (runRef.current) return;
    flushDraft();
    writeLocal(NEW_DRAFT_KEY, "");
    show(null, [], EMPTY_DRAFT);
  }, [flushDraft, show]);

  /** One event from the engine, whichever call it arrived on. */
  const onEvent = useCallback(
    (ev: ImageEvent) => {
      if (ev.type === "started") {
        setRun(freshRun(ev.id, ev.request, ev.startedAt));
        busyRef.current(true);
        void refreshSessions();
      } else if (ev.type === "engine") {
        setRun((r) => (r ? apply(r, ev.event) : r));
      } else if (ev.type === "done") {
        const rec = ev.record;
        if (rec) {
          if (rec.sessionId === sessionRef.current) {
            setRounds((rs) => [...rs.filter((x) => x.id !== rec.id), rec]);
            // Multi-turn editing: the next prompt edits this picture, unless
            // another one was picked while it was drawn.
            if (chainRef.current && rec.images[0]) {
              setDraft((d) => (d.reference ? d : { ...d, reference: { path: rec.images[0].path, parentId: rec.id } }));
            }
          }
          void refreshSessions();
        } else if (ev.cancelled) {
          const req = runRef.current?.request;
          if (req && (req.sessionId ?? null) === sessionRef.current) setDraft((d) => restored(d, req));
          notifyRef.current("warn", stoppedText);
        }
        setRun(null);
        busyRef.current(false);
      } else if (ev.type === "error") {
        setRun(null);
        busyRef.current(false);
        notifyRef.current("error", ev.message.slice(0, 300));
      }
    },
    [stoppedText, refreshSessions],
  );

  // Entering the studio the first time: the session list, the session open
  // last (or the new session's draft), and a round that was already being
  // drawn when this page loaded (the webview can reload under one).
  useEffect(() => {
    if (!active) return;
    void refreshSessions();
    if (!started.current) {
      started.current = true;
      void (async () => {
        const live = await imageAttach(onEvent).catch(() => null);
        const want = live?.request.sessionId || readLocal(LAST_SESSION_KEY);
        const data = want ? await imageSessionGet(want).catch(() => null) : null;
        if (data) show(data.session.id, data.records, parseDraft(data.draft));
        else show(null, [], parseDraft(readLocal(NEW_DRAFT_KEY)));
        if (live) {
          let r = freshRun(live.id, live.request, live.startedAt);
          for (const ev of live.events) r = apply(r, ev);
          setRun(r);
          busyRef.current(true);
        }
      })();
    } else {
      // Back from a chat model: a round may have been drawn meanwhile.
      const id = sessionRef.current;
      if (id && !runRef.current) {
        void imageSessionGet(id)
          .then((data) => data && setRounds(data.records))
          .catch(() => {});
      }
    }
  }, [active, refreshSessions, onEvent, show]);

  /** Draw a round in the open session, starting the session if it is new. */
  const generate = useCallback(
    async (request: ImageRequest, parentId?: string | null) => {
      if (runRef.current) return;
      let id = sessionRef.current;
      try {
        if (!id) {
          id = uid();
          await imageSessionSave(id, convTitle(request.prompt));
          writeLocal(NEW_DRAFT_KEY, "");
          setSessionId(id);
          sessionRef.current = id;
          writeLocal(LAST_SESSION_KEY, id);
        }
        // Sent, like a chat message: the composer empties.
        pendingDraft.current = null;
        setDraft((d) => ({ ...d, prompt: "", reference: null }));
        const sent = { ...request, sessionId: id, parentId: parentId ?? null };
        try {
          await imageGenerate(sent, onEvent);
        } catch (e) {
          if (sessionRef.current === id) setDraft((d) => restored(d, sent));
          throw e;
        }
      } catch (e) {
        // The error already arrived as an event (and was shown); this is the
        // same failure returned by the call. Log it for the report.
        void logAppError("image-generate", String((e as Error)?.message ?? e)).catch(() => {});
        setRun(null);
        busyRef.current(false);
      }
    },
    [onEvent],
  );

  const cancel = useCallback(async (mode: "all" | "after") => {
    setRun((r) => (r ? { ...r, stopping: mode } : r));
    await imageCancel(mode === "after" ? "after_current" : "all").catch(console.error);
  }, []);

  /** Delete one round of the open session, and its pictures. */
  const removeRound = useCallback(async (id: string) => {
    await imageGenerationDelete(id, true);
    setRounds((rs) => rs.filter((r) => r.id !== id));
    setDraft((d) => (d.reference?.parentId === id ? { ...d, reference: null } : d));
  }, []);

  /** Delete a session and its pictures — stopping its round if one is being
   *  drawn, as deleting a chat stops its reply. */
  const deleteSession = useCallback(
    async (id: string) => {
      if (runRef.current?.request.sessionId === id) await imageCancel("all").catch(() => {});
      await imageSessionDelete(id, true);
      if (id === sessionRef.current) {
        pendingDraft.current = null;
        show(null, [], EMPTY_DRAFT);
      }
      await refreshSessions();
    },
    [refreshSessions, show],
  );

  const togglePin = useCallback(
    async (s: ImageSession) => {
      await imageSessionSetPinned(s.id, !s.pinned);
      await refreshSessions();
    },
    [refreshSessions],
  );

  const rename = useCallback(
    async (id: string, title: string) => {
      await imageSessionRename(id, title);
      await refreshSessions();
    },
    [refreshSessions],
  );

  /** Settings → Data cleared every session: start over on a new one. */
  const cleared = useCallback(() => {
    pendingDraft.current = null;
    writeLocal(NEW_DRAFT_KEY, "");
    show(null, [], EMPTY_DRAFT);
    setSessions([]);
  }, [show]);

  const setPrompt = useCallback((prompt: string) => setDraft((d) => ({ ...d, prompt })), []);
  const setNegative = useCallback((negative: string | null) => setDraft((d) => ({ ...d, negative })), []);
  const setReference = useCallback((reference: ImageRef | null) => setDraft((d) => ({ ...d, reference })), []);

  return {
    sessions,
    sessionId,
    session: sessions.find((s) => s.id === sessionId) ?? null,
    rounds,
    run,
    /** The round being drawn belongs to the open session. */
    runHere: !!run && (run.request.sessionId ?? null) === sessionId,
    prompt: draft.prompt,
    setPrompt,
    /** null = the default negative prompt from Settings. */
    negative: draft.negative,
    setNegative,
    reference: draft.reference,
    setReference,
    refreshSessions,
    openSession,
    startNew,
    generate,
    cancel,
    removeRound,
    deleteSession,
    togglePin,
    rename,
    cleared,
  };
}

export type ImageStudioState = ReturnType<typeof useImageStudio>;
