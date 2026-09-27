// The music studio's state, shaped like the image studio's: a list of
// sessions, the one open with its rounds, and the piece being made. Shared by
// the sidebar (sessions) and the studio (thread and composer), so it lives
// above both — in App — as one hook. It outlives a model switch: loading
// another music model carries on in the session that is open.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  logAppError,
  musicAttach,
  musicCancel,
  musicGenerate,
  musicSessionDelete,
  musicSessionDraft,
  musicSessionGet,
  musicSessionList,
  musicSessionRename,
  musicSessionSave,
  musicSessionSetPinned,
  musicTrackDelete,
  type MusicEdit,
  type MusicEngineEvent,
  type MusicEvent,
  type MusicRecord,
  type MusicRequest,
  type MusicSession,
} from "./ipc";
import { convTitle } from "./fmt";
import { EMPTY_DRAFT, type MusicDraft, type MusicProgress } from "./musicGen";

/** A piece being made, as the studio shows it. */
export interface MusicRun {
  id: string;
  request: MusicRequest;
  startedAt: number;
  /** Stages begun so far, in order. */
  stages: string[];
  progress: MusicProgress;
  seed: number | null;
  /** A stop was asked for. */
  stopping: boolean;
}

/** The session open last, reopened on the next start. */
const LAST_SESSION_KEY = "chaty.musicSession";
/** The draft of a session not yet started. */
const NEW_DRAFT_KEY = "chaty.musicDraft";

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
export function parseMusicDraft(raw: string): MusicDraft {
  try {
    const d = JSON.parse(raw) as Partial<MusicDraft>;
    return {
      prompt: typeof d.prompt === "string" ? d.prompt : "",
      lyrics: typeof d.lyrics === "string" ? d.lyrics : "",
      mode: d.mode === "instrumental" ? "instrumental" : "song",
      scorePath: typeof d.scorePath === "string" ? d.scorePath : null,
      edit: d.edit && typeof d.edit.parentId === "string" && typeof d.edit.kind === "string" ? d.edit : null,
    };
  } catch {
    return EMPTY_DRAFT;
  }
}

function draftJson(d: MusicDraft): string {
  return d.prompt || d.lyrics || d.mode !== "song" || d.scorePath || d.edit ? JSON.stringify(d) : "";
}

function freshRun(id: string, request: MusicRequest, startedAt: number): MusicRun {
  return {
    id,
    request,
    startedAt,
    stages: [],
    progress: { stage: "prepare", done: 0, total: 0, seconds: null },
    seed: null,
    stopping: false,
  };
}

export function applyMusic(run: MusicRun, ev: MusicEngineEvent): MusicRun {
  switch (ev.type) {
    case "stage":
      return {
        ...run,
        stages: run.stages.includes(ev.stage) ? run.stages : [...run.stages, ev.stage],
        progress: { ...run.progress, stage: ev.stage, done: 0, total: 0 },
        seed: ev.seed ?? run.seed,
      };
    case "progress":
      return {
        ...run,
        stages: run.stages.includes(ev.stage) ? run.stages : [...run.stages, ev.stage],
        progress: { ...run.progress, stage: ev.stage, done: ev.done, total: ev.total },
      };
    case "info":
      return ev.key === "seconds" ? { ...run, progress: { ...run.progress, seconds: ev.value } } : run;
    case "audio":
      return run;
  }
  return run;
}

/** A round that made nothing (stopped, or failed) gives its words back to an
 *  empty composer. */
function restored(d: MusicDraft, req: MusicRequest): MusicDraft {
  if (d.prompt.trim() || d.lyrics.trim()) return d;
  return {
    ...d,
    prompt: req.prompt,
    lyrics: req.lyrics,
    mode: req.instrumental ? "instrumental" : d.mode,
    edit: d.edit ?? req.edit ?? null,
  };
}

export function useMusicStudio({
  active,
  onBusy,
  notify,
  stoppedText,
}: {
  /** A music model is loaded — the studio is on screen. */
  active: boolean;
  onBusy: (busy: boolean) => void;
  notify: (kind: "warn" | "error", text: string) => void;
  stoppedText: string;
}) {
  const [sessions, setSessions] = useState<MusicSession[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [rounds, setRounds] = useState<MusicRecord[]>([]);
  const [run, setRun] = useState<MusicRun | null>(null);
  const [draft, setDraft] = useState<MusicDraft>(EMPTY_DRAFT);
  /** The round just finished, for the player to start. */
  const [fresh, setFresh] = useState<string | null>(null);
  const runRef = useRef<MusicRun | null>(null);
  runRef.current = run;
  const sessionRef = useRef<string | null>(null);
  sessionRef.current = sessionId;
  const busyRef = useRef(onBusy);
  busyRef.current = onBusy;
  const notifyRef = useRef(notify);
  notifyRef.current = notify;
  const pendingDraft = useRef<{ session: string | null; json: string } | null>(null);
  const started = useRef(false);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await musicSessionList());
    } catch (e) {
      console.error(e);
    }
  }, []);

  const writeDraft = useCallback((session: string | null, json: string) => {
    if (session) void musicSessionDraft(session, json).catch(() => {});
    else writeLocal(NEW_DRAFT_KEY, json);
  }, []);

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

  const show = useCallback((id: string | null, recs: MusicRecord[], d: MusicDraft) => {
    setSessionId(id);
    setRounds(recs);
    setDraft(d);
    writeLocal(LAST_SESSION_KEY, id ?? "");
  }, []);

  const openSession = useCallback(
    async (id: string) => {
      if (runRef.current || id === sessionRef.current) return;
      flushDraft();
      try {
        const data = await musicSessionGet(id);
        if (!data) {
          void refreshSessions();
          return;
        }
        show(id, data.records, parseMusicDraft(data.draft));
      } catch (e) {
        console.error(e);
      }
    },
    [flushDraft, refreshSessions, show],
  );

  const startNew = useCallback(() => {
    if (runRef.current) return;
    flushDraft();
    writeLocal(NEW_DRAFT_KEY, "");
    show(null, [], EMPTY_DRAFT);
  }, [flushDraft, show]);

  const onEvent = useCallback(
    (ev: MusicEvent) => {
      if (ev.type === "started") {
        setRun(freshRun(ev.id, ev.request, ev.startedAt));
        busyRef.current(true);
        void refreshSessions();
      } else if (ev.type === "engine") {
        setRun((r) => (r ? applyMusic(r, ev.event) : r));
      } else if (ev.type === "done") {
        const rec = ev.record;
        if (rec) {
          if (rec.sessionId === sessionRef.current) {
            setRounds((rs) => [...rs.filter((x) => x.id !== rec.id), rec]);
            setFresh(rec.id);
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
        const req = runRef.current?.request;
        if (req && (req.sessionId ?? null) === sessionRef.current) setDraft((d) => restored(d, req));
        setRun(null);
        busyRef.current(false);
        notifyRef.current("error", ev.message.slice(0, 400));
      }
    },
    [stoppedText, refreshSessions],
  );

  // Entering the studio the first time: the session list, the session open
  // last (or the new session's draft), and a piece that was already being made
  // when this page loaded.
  useEffect(() => {
    if (!active) return;
    void refreshSessions();
    if (!started.current) {
      started.current = true;
      void (async () => {
        const live = await musicAttach(onEvent).catch(() => null);
        const want = live?.request.sessionId || readLocal(LAST_SESSION_KEY);
        const data = want ? await musicSessionGet(want).catch(() => null) : null;
        if (data) show(data.session.id, data.records, parseMusicDraft(data.draft));
        else show(null, [], parseMusicDraft(readLocal(NEW_DRAFT_KEY)));
        if (live) {
          let r = freshRun(live.id, live.request, live.startedAt);
          for (const ev of live.events) r = applyMusic(r, ev);
          setRun(r);
          busyRef.current(true);
        }
      })();
    } else {
      const id = sessionRef.current;
      if (id && !runRef.current) {
        void musicSessionGet(id)
          .then((data) => data && setRounds(data.records))
          .catch(() => {});
      }
    }
  }, [active, refreshSessions, onEvent, show]);

  /** Make a piece in the open session, starting the session if it is new. */
  const generate = useCallback(
    async (request: MusicRequest) => {
      if (runRef.current) return;
      let id = sessionRef.current;
      try {
        if (!id) {
          id = uid();
          await musicSessionSave(id, convTitle(request.prompt || request.lyrics.split("\n").find((l) => l.trim() && !/^\[/.test(l.trim())) || "♪"));
          writeLocal(NEW_DRAFT_KEY, "");
          setSessionId(id);
          sessionRef.current = id;
          writeLocal(LAST_SESSION_KEY, id);
        }
        // Sent, like a chat message: the words stay for the next variation —
        // a song is refined more than it is replaced — but the edit is spent.
        pendingDraft.current = null;
        setDraft((d) => ({ ...d, edit: null }));
        await musicGenerate({ ...request, sessionId: id }, onEvent);
      } catch (e) {
        void logAppError("music-generate", String((e as Error)?.message ?? e)).catch(() => {});
        setRun(null);
        busyRef.current(false);
      }
    },
    [onEvent],
  );

  const cancel = useCallback(async () => {
    setRun((r) => (r ? { ...r, stopping: true } : r));
    await musicCancel().catch(console.error);
  }, []);

  const removeRound = useCallback(async (id: string) => {
    await musicTrackDelete(id, true);
    setRounds((rs) => rs.filter((r) => r.id !== id));
    setDraft((d) => (d.edit?.parentId === id ? { ...d, edit: null } : d));
  }, []);

  const deleteSession = useCallback(
    async (id: string) => {
      if (runRef.current?.request.sessionId === id) await musicCancel().catch(() => {});
      await musicSessionDelete(id, true);
      if (id === sessionRef.current) {
        pendingDraft.current = null;
        show(null, [], EMPTY_DRAFT);
      }
      await refreshSessions();
    },
    [refreshSessions, show],
  );

  const togglePin = useCallback(
    async (s: MusicSession) => {
      await musicSessionSetPinned(s.id, !s.pinned);
      await refreshSessions();
    },
    [refreshSessions],
  );

  const rename = useCallback(
    async (id: string, title: string) => {
      await musicSessionRename(id, title);
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

  const patchDraft = useCallback((patch: Partial<MusicDraft>) => setDraft((d) => ({ ...d, ...patch })), []);
  const setEdit = useCallback((edit: MusicEdit | null) => setDraft((d) => ({ ...d, edit })), []);

  return {
    sessions,
    sessionId,
    session: sessions.find((s) => s.id === sessionId) ?? null,
    rounds,
    run,
    /** The piece being made belongs to the open session. */
    runHere: !!run && (run.request.sessionId ?? null) === sessionId,
    draft,
    patchDraft,
    setDraft,
    setEdit,
    /** The round just finished (the player starts it), and its reset. */
    fresh,
    clearFresh: () => setFresh(null),
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

export type MusicStudioState = ReturnType<typeof useMusicStudio>;
