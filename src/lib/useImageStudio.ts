// The image studio's state: the history of generations, the one on screen,
// and the one being drawn. Shared by the sidebar (history) and the studio
// (canvas and composer), so it lives above both — in App — as one hook.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  imageAttach,
  imageCancel,
  imageGenerate,
  imageHistoryClear,
  imageHistoryDelete,
  imageHistoryList,
  logAppError,
  type ImageEvent,
  type ImageItem,
  type ImageRecord,
  type ImageRequest,
  type SdEvent,
} from "./ipc";
import { percent, smooth, type GenState } from "./imageGen";

/** A generation in progress, as the studio draws it. */
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
}

/** A picture to start from (img2img) or to edit (editing models). */
export interface ImageRef {
  path: string;
}

const FRESH: GenState = { stage: "encode", index: 0, count: 1, step: 0, steps: 0, secsPerStep: 0 };

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
  }
}

export function useImageStudio({
  active,
  onBusy,
  notify,
  stoppedText,
}: {
  /** An image model is loaded — the studio is on screen. */
  active: boolean;
  onBusy: (busy: boolean) => void;
  notify: (kind: "warn" | "error", text: string) => void;
  /** Toast for a generation stopped before it finished a picture. */
  stoppedText: string;
}) {
  const [history, setHistory] = useState<ImageRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [run, setRun] = useState<ImageRun | null>(null);
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState<string | null>(null);
  const [reference, setReference] = useState<ImageRef | null>(null);
  const runRef = useRef<ImageRun | null>(null);
  runRef.current = run;
  const busyRef = useRef(onBusy);
  busyRef.current = onBusy;
  const notifyRef = useRef(notify);
  notifyRef.current = notify;

  const refresh = useCallback(async () => {
    try {
      setHistory(await imageHistoryList());
    } catch (e) {
      console.error(e);
    }
  }, []);

  /** One event from the engine, whichever call it arrived on. */
  const onEvent = useCallback(
    (ev: ImageEvent) => {
      if (ev.type === "started") {
        setRun({
          id: ev.id,
          request: ev.request,
          startedAt: ev.startedAt,
          gen: { ...FRESH, count: Math.max(1, ev.request.batchCount) },
          pct: 0,
          seed: null,
          preview: null,
          images: [],
          stopping: "",
        });
        setSelectedId(null);
        busyRef.current(true);
      } else if (ev.type === "engine") {
        setRun((r) => (r ? apply(r, ev.event) : r));
      } else if (ev.type === "done") {
        const rec = ev.record;
        if (rec) {
          setHistory((h) => [rec, ...h.filter((x) => x.id !== rec.id)]);
          setSelectedId(rec.id);
        } else if (ev.cancelled) {
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
    [stoppedText],
  );

  // Entering the studio: read the history, and rejoin a generation that was
  // already running when this page loaded (the webview can reload under one).
  useEffect(() => {
    if (!active) return;
    void refresh();
    void imageAttach(onEvent)
      .then((live) => {
        if (!live) return;
        let r: ImageRun = {
          id: live.id,
          request: live.request,
          startedAt: live.startedAt,
          gen: { ...FRESH, count: Math.max(1, live.request.batchCount) },
          pct: 0,
          seed: null,
          preview: null,
          images: [],
          stopping: "",
        };
        for (const ev of live.events) r = apply(r, ev);
        setRun(r);
        setSelectedId(null);
        busyRef.current(true);
      })
      .catch(() => {});
  }, [active, refresh, onEvent]);

  const generate = useCallback(
    async (request: ImageRequest) => {
      if (runRef.current) return;
      try {
        await imageGenerate(request, onEvent);
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

  const remove = useCallback(async (id: string, deleteFiles: boolean) => {
    await imageHistoryDelete(id, deleteFiles);
    setHistory((h) => h.filter((r) => r.id !== id));
    setSelectedId((cur) => (cur === id ? null : cur));
  }, []);

  const clear = useCallback(async (deleteFiles: boolean) => {
    await imageHistoryClear(deleteFiles);
    setHistory([]);
    setSelectedId(null);
  }, []);

  /** A blank canvas: nothing selected, an empty prompt. */
  const startNew = useCallback(() => {
    setSelectedId(null);
    setPrompt("");
    setReference(null);
  }, []);

  return {
    history,
    selected: history.find((r) => r.id === selectedId) ?? null,
    selectedId,
    select: setSelectedId,
    run,
    prompt,
    setPrompt,
    /** null = the default negative prompt from Settings. */
    negative,
    setNegative,
    reference,
    setReference,
    refresh,
    generate,
    cancel,
    remove,
    clear,
    startNew,
  };
}

export type ImageStudioState = ReturnType<typeof useImageStudio>;
