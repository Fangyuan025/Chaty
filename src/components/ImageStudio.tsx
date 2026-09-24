import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { open as openDialog } from "@tauri-apps/plugin-dialog";

import { useI18n, type TKey } from "../lib/i18n";
import { watchContentHeight } from "../lib/autoGrow";
import {
  imageCopy,
  openExternal,
  saveImageAs,
  type ImageItem,
  type ImageRecord,
  type ModelInfo,
} from "../lib/ipc";
import {
  ASPECTS,
  BASE_SIZES,
  SAMPLERS,
  SCHEDULERS,
  buildRequest,
  effective,
  etaSeconds,
  fmtDuration,
  type ImageSettings,
} from "../lib/imageGen";
import type { ImageRun, ImageStudioState } from "../lib/useImageStudio";
import { Icon } from "./Icon";
import { ImageThumb } from "./ImageThumb";
import { useConfirm } from "./ConfirmModal";

/** Ideas for an empty canvas — written to show off what these models are
 *  good at (text in the picture, light, texture). */
const IDEAS_ZH = [
  "雨夜街角的霓虹灯招牌，写着「深夜食堂」，湿漉漉的路面倒映着灯光，电影感",
  "水彩风格的橘猫趴在洒满阳光的窗台上，旁边一盆薄荷",
  "极简主义海报：一座白色灯塔，蓝色渐变天空，大号无衬线标题 “CHATY”",
  "等距视角的微缩书房，暖色台灯，木质书架，细节丰富的 3D 渲染",
];
const IDEAS_EN = [
  "A neon sign that reads \"LATE NIGHT DINER\" on a rainy street corner, reflections on wet pavement, cinematic",
  "Watercolor of a ginger cat asleep on a sunny windowsill beside a pot of mint",
  "Minimalist poster: a white lighthouse under a blue gradient sky, large sans-serif title \"CHATY\"",
  "Isometric miniature study room, warm desk lamp, wooden bookshelves, detailed 3D render",
];

const STAGE_KEY: Record<string, TKey> = {
  encode: "imgStageEncode",
  weights: "imgStageWeights",
  sample: "imgStageSample",
  decode: "imgStageDecode",
};

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

/** One finished picture with its hover actions. */
function Picture({
  item,
  onPreview,
  onUseAsRef,
  onReuseSeed,
  notify,
  big,
}: {
  item: ImageItem;
  onPreview: (p: string) => void;
  onUseAsRef?: (p: string) => void;
  onReuseSeed: (seed: number) => void;
  notify: (kind: "warn" | "error", text: string) => void;
  big: boolean;
}) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  return (
    <div className={`is-pic ${big ? "big" : ""}`} style={{ aspectRatio: `${item.width} / ${item.height}` }}>
      <ImageThumb path={item.path} size={4096} onOpen={() => onPreview(item.path)} className="is-pic-img" />
      <div className="is-pic-bar">
        <span className="is-pic-meta">
          {item.width}×{item.height} · {t("imgSeed")} {item.seed}
        </span>
        <span className="is-pic-actions">
          <button title={t("imgActZoom")} onClick={() => onPreview(item.path)}>
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
            </svg>
          </button>
          <button
            title={t("imgActSave")}
            onClick={() => void saveImageAs(item.path, basename(item.path)).catch((e) => notify("error", String(e)))}
          >
            <Icon name="download" size={14} />
          </button>
          <button
            title={copied ? t("imgCopied") : t("imgActCopy")}
            onClick={() =>
              void imageCopy(item.path)
                .then(() => {
                  setCopied(true);
                  setTimeout(() => setCopied(false), 1400);
                })
                .catch((e) => notify("error", String(e)))
            }
          >
            {copied ? (
              <Icon name="check" size={14} strokeWidth={2.2} />
            ) : (
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <rect x="9" y="9" width="11" height="11" rx="2" />
                <path d="M5 15V5a2 2 0 0 1 2-2h8" strokeLinecap="round" />
              </svg>
            )}
          </button>
          <button title={t("imgActReveal")} onClick={() => void openExternal(dirname(item.path)).catch(console.error)}>
            <Icon name="folder" size={14} />
          </button>
          {onUseAsRef && (
            <button title={t("imgActUseRef")} onClick={() => onUseAsRef(item.path)}>
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <path d="M21 15l-5-5L5 21" />
              </svg>
            </button>
          )}
          <button title={t("imgActReuseSeed")} onClick={() => onReuseSeed(item.seed)}>
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="4" y="4" width="16" height="16" rx="3" />
              <circle cx="9" cy="9" r="1" fill="currentColor" />
              <circle cx="15" cy="15" r="1" fill="currentColor" />
            </svg>
          </button>
        </span>
      </div>
    </div>
  );
}

function PictureGrid({ items, ...rest }: { items: ImageItem[] } & Omit<Parameters<typeof Picture>[0], "item" | "big">) {
  return (
    <div className={`is-grid n${Math.min(items.length, 4)}`}>
      {items.map((it) => (
        <Picture key={it.path} item={it} big={items.length === 1} {...rest} />
      ))}
    </div>
  );
}

/** The chips under a generation: what it ran with. */
function ParamChips({ rec }: { rec: ImageRecord }) {
  const { t } = useI18n();
  const p = rec.params;
  const seeds = rec.images.map((i) => i.seed);
  const chips = [
    p.width && p.height ? `${p.width}×${p.height}` : null,
    p.steps ? t("imgStepsN", { n: p.steps }) : null,
    p.cfgScale != null ? `CFG ${p.cfgScale}` : null,
    p.sampler || null,
    seeds.length ? `${t("imgSeed")} ${seeds.length > 1 ? `${seeds[0]}…${seeds[seeds.length - 1]}` : seeds[0]}` : null,
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

/** The generation being drawn: percentage, phase, live preview. */
function RunView({ run, onStop }: { run: ImageRun; onStop: (mode: "all" | "after") => void }) {
  const { t } = useI18n();
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(id);
  }, []);
  const g = run.gen;
  const pct = Math.round(run.pct);
  const eta = etaSeconds(g);
  const req = run.request;
  const stage = t(STAGE_KEY[g.stage] ?? "imgStageEncode");
  const detail: string[] = [];
  if (g.count > 1) detail.push(t("imgPicN", { i: Math.min(g.index + 1, g.count), n: g.count }));
  if ((g.stage === "sample" || g.stage === "decode" || g.stage === "weights") && g.steps > 0)
    detail.push(g.stage === "weights" ? `${Math.round((g.step / g.steps) * 100)}%` : `${g.step}/${g.steps}`);
  if (g.stage === "sample" && g.secsPerStep > 0)
    detail.push(g.secsPerStep >= 1 ? `${g.secsPerStep.toFixed(1)} ${t("imgSecPerStep")}` : `${(1 / g.secsPerStep).toFixed(1)} ${t("imgStepPerSec")}`);
  detail.push(t("imgElapsed", { t: fmtDuration(now - run.startedAt) }));
  if (eta != null) detail.push(t("imgEta", { t: fmtDuration(eta * 1000) }));

  return (
    <div className="is-run">
      <div className="is-prompt-line">{req.prompt}</div>
      <div className="is-run-stage" style={{ aspectRatio: `${req.width} / ${req.height}` }}>
        {run.preview ? <img className="is-run-preview" src={run.preview} alt="" /> : <div className="is-run-ph" />}
        <div className="is-run-pct">
          <span className="is-run-num">{pct}</span>
          <span className="is-run-sign">%</span>
        </div>
      </div>
      <div className="is-run-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="is-run-fill" style={{ width: `${Math.max(1.5, run.pct)}%` }} />
      </div>
      <div className="is-run-info">
        <span className="is-run-label">
          <span className="is-dot" /> {run.stopping ? t("imgStopping") : stage}
        </span>
        <span className="is-run-detail">{detail.join(" · ")}</span>
      </div>
      <div className="is-run-actions">
        {g.count > 1 && g.index + 1 < g.count && (
          <button className="is-btn" disabled={!!run.stopping} onClick={() => onStop("after")}>
            {t("imgStopAfter")}
          </button>
        )}
        <button className="is-btn danger" disabled={run.stopping === "all"} onClick={() => onStop("all")}>
          {t("imgStop")}
        </button>
      </div>
    </div>
  );
}

export function ImageStudio({
  model,
  studio,
  settings,
  onSettings,
  onOpenSettings,
  onPreview,
  notify,
  sendKey,
}: {
  model: ModelInfo;
  studio: ImageStudioState;
  settings: ImageSettings;
  onSettings: (patch: Partial<ImageSettings>) => void;
  /** "More settings" — the image tab of the settings panel. */
  onOpenSettings: () => void;
  onPreview: (path: string) => void;
  notify: (kind: "warn" | "error", text: string) => void;
  sendKey: "enter" | "modEnter";
}) {
  const { t, lang } = useI18n();
  const confirm = useConfirm();
  const info = model.image!;
  const d = info.defaults;
  const eff = useMemo(() => effective(settings, d), [settings, d]);
  const [pop, setPop] = useState<"" | "size" | "params">("");
  const [showNeg, setShowNeg] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const run = studio.run;
  const busy = !!run;
  const negative = studio.negative ?? settings.imgNegative;
  const edits = info.edits;

  // The prompt box grows with its content, like the chat composer.
  useLayoutEffect(() => {
    const el = inputRef.current;
    const row = el?.parentElement;
    if (!el || !row) return;
    return watchContentHeight(el, row, 200);
  }, [studio.prompt]);

  // A new result or a new run starts at the top of the canvas.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [studio.selectedId, run?.id]);

  const go = () => {
    const prompt = studio.prompt.trim();
    if (!prompt || busy) return;
    void studio.generate(
      buildRequest(prompt, negative, settings, d, studio.reference ? { path: studio.reference.path, edit: edits } : null),
    );
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== "Enter" || e.nativeEvent.isComposing) return;
    if (sendKey === "modEnter") {
      if (e.metaKey || e.ctrlKey) {
        e.preventDefault();
        go();
      }
    } else if (!e.shiftKey) {
      e.preventDefault();
      go();
    }
  };

  const pickReference = async () => {
    const picked = await openDialog({
      multiple: false,
      filters: [{ name: t("imgImages"), extensions: ["png", "jpg", "jpeg", "webp", "bmp"] }],
    });
    if (typeof picked === "string") studio.setReference({ path: picked });
  };

  const reuse = (rec: ImageRecord) => {
    const p = rec.params;
    studio.setPrompt(rec.prompt);
    studio.setNegative(rec.negativePrompt || null);
    onSettings({
      imgSteps: p.steps ?? settings.imgSteps,
      imgCfg: p.cfgScale ?? settings.imgCfg,
      imgSampler: p.sampler ?? settings.imgSampler,
      imgScheduler: p.scheduler ?? settings.imgScheduler,
      imgBatch: p.batchCount ?? settings.imgBatch,
      imgAspect: "custom",
      imgCustomW: p.width ?? settings.imgCustomW,
      imgCustomH: p.height ?? settings.imgCustomH,
    });
    inputRef.current?.focus();
  };

  const again = (rec: ImageRecord) => {
    if (busy) return;
    void studio.generate({
      ...buildRequest(rec.prompt, rec.negativePrompt, settings, d, null),
      ...rec.params,
      prompt: rec.prompt,
      negativePrompt: rec.negativePrompt,
      // Again = the same request with fresh randomness.
      seed: -1,
      outDir: settings.imgOutputDir.trim() || null,
    });
  };

  const del = async (rec: ImageRecord) => {
    const ok = await confirm({
      title: t("imgDeleteTitle"),
      message: t("imgDeleteConfirm"),
      confirmLabel: t("confirmDelete"),
      danger: true,
    });
    if (ok) await studio.remove(rec.id, true).catch((e) => notify("error", String(e)));
  };

  const pictureProps = {
    onPreview,
    onUseAsRef: (p: string) => {
      studio.setReference({ path: p });
      inputRef.current?.focus();
    },
    onReuseSeed: (seed: number) => {
      onSettings({ imgSeedLock: true, imgSeed: seed });
    },
    notify,
  };

  const rec = studio.selected;
  const sizeLabel = `${eff.width}×${eff.height}`;
  const ideas = lang === "zh" ? IDEAS_ZH : IDEAS_EN;

  return (
    <div className="is-studio">
      <div className="is-canvas" ref={scrollRef}>
        {run ? (
          <>
            <RunView run={run} onStop={(m) => void studio.cancel(m)} />
            {run.images.length > 0 && <PictureGrid items={run.images} {...pictureProps} />}
          </>
        ) : rec ? (
          <div className="is-record">
            <div className="is-prompt-line selectable">{rec.prompt}</div>
            {rec.negativePrompt && (
              <div className="is-neg-line">
                <span>{t("imgNegative")}</span> {rec.negativePrompt}
              </div>
            )}
            <ParamChips rec={rec} />
            <PictureGrid items={rec.images} {...pictureProps} />
            <div className="is-record-actions">
              <button className="is-btn" onClick={() => again(rec)} disabled={busy}>
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5" />
                </svg>
                {t("imgAgain")}
              </button>
              <button className="is-btn" onClick={() => reuse(rec)}>
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M4 20h4L19 9l-4-4L4 16zM14 6l4 4" />
                </svg>
                {t("imgReuse")}
              </button>
              <button className="is-btn danger-text" onClick={() => void del(rec)}>
                <Icon name="x" size={12} strokeWidth={2.2} />
                {t("imgDelete")}
              </button>
            </div>
          </div>
        ) : (
          <div className="empty is-empty">
            <div className="empty-hero">
              <div className="empty-greeting">{t("imgHero")}</div>
              <div className="empty-sub">
                {t("imgHeroSub", { model: info.engineVersion || info.familyName, size: sizeLabel })}
              </div>
            </div>
            <div className="suggestions">
              {ideas.map((s) => (
                <button
                  key={s}
                  className="suggestion"
                  onClick={() => {
                    studio.setPrompt(s);
                    inputRef.current?.focus();
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <footer className="composer is-composer">
        {(studio.reference || (showNeg && d.negativePrompt)) && (
          <div className="is-extras">
            {studio.reference && (
              <div className="is-ref">
                <ImageThumb path={studio.reference.path} size={40} onOpen={() => onPreview(studio.reference!.path)} />
                <div className="is-ref-text">
                  <span className="is-ref-title">{edits ? t("imgRefEdit") : t("imgRefInit")}</span>
                  {!edits && (
                    <label className="is-ref-strength" title={t("imgStrengthHint")}>
                      {t("imgStrength")} <b>{settings.imgStrength.toFixed(2)}</b>
                      <input
                        type="range"
                        min={0.05}
                        max={1}
                        step={0.05}
                        value={settings.imgStrength}
                        onChange={(e) => onSettings({ imgStrength: Number(e.target.value) })}
                      />
                    </label>
                  )}
                </div>
                <button className="attach-remove" title={t("imgRefRemove")} onClick={() => studio.setReference(null)}>
                  <Icon name="x" size={11} strokeWidth={2.2} />
                </button>
              </div>
            )}
            {showNeg && d.negativePrompt && (
              <label className="is-neg">
                <span>{t("imgNegative")}</span>
                <input
                  type="text"
                  value={negative}
                  placeholder={t("imgNegativePh")}
                  onChange={(e) => studio.setNegative(e.target.value)}
                />
              </label>
            )}
          </div>
        )}
        <div className="input-row is-input">
          <button className="tool-toggle" title={t("imgRefAdd")} onClick={() => void pickReference()}>
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2.5" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <path d="M21 15l-5-5L5 21" />
            </svg>
          </button>
          <textarea
            ref={inputRef}
            value={studio.prompt}
            onChange={(e) => studio.setPrompt(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={sendKey === "modEnter" ? t("imgPromptPhMod") : t("imgPromptPh")}
            rows={1}
          />
          {busy ? (
            <button className="send-btn stop" onClick={() => void studio.cancel("all")} title={t("imgStop")}>
              <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="3" fill="currentColor" />
              </svg>
            </button>
          ) : (
            <button className="send-btn is-go" onClick={go} disabled={!studio.prompt.trim()} title={t("imgGenerate")}>
              <svg width="19" height="19" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 2.5l2.2 6.3L20.5 11l-6.3 2.2L12 19.5l-2.2-6.3L3.5 11l6.3-2.2z" />
              </svg>
            </button>
          )}
        </div>

        <div className="is-bar">
          <div className="is-bar-item">
            <button className={`is-bar-btn ${pop === "size" ? "on" : ""}`} onClick={() => setPop(pop === "size" ? "" : "size")}>
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <rect x="4" y="6" width="16" height="12" rx="2" />
              </svg>
              {settings.imgAspect === "custom" ? t("imgAspectCustom") : settings.imgAspect} · {sizeLabel}
            </button>
            <Pop open={pop === "size"} onClose={() => setPop("")} className="is-pop-size">
              <div className="is-pop-title">{t("imgAspect")}</div>
              <div className="is-aspects">
                {ASPECTS.map((a) => {
                  const [w, h] = a.split(":").map(Number);
                  const k = 16 / Math.max(w, h);
                  return (
                    <button key={a} className={settings.imgAspect === a ? "on" : ""} onClick={() => onSettings({ imgAspect: a })}>
                      <span className="is-aspect-box" style={{ width: w * k, height: h * k }} />
                      {a}
                    </button>
                  );
                })}
                <button className={settings.imgAspect === "custom" ? "on" : ""} onClick={() => onSettings({ imgAspect: "custom", imgCustomW: eff.width, imgCustomH: eff.height })}>
                  <span className="is-aspect-box dashed" style={{ width: 14, height: 14 }} />
                  {t("imgAspectCustom")}
                </button>
              </div>
              {settings.imgAspect === "custom" ? (
                <div className="is-custom-size">
                  <input type="number" min={d.align} max={4096} step={d.align} value={settings.imgCustomW} onChange={(e) => onSettings({ imgCustomW: Number(e.target.value) })} />
                  <span>×</span>
                  <input type="number" min={d.align} max={4096} step={d.align} value={settings.imgCustomH} onChange={(e) => onSettings({ imgCustomH: Number(e.target.value) })} />
                  <small>{t("imgAlignHint", { n: d.align })}</small>
                </div>
              ) : (
                <>
                  <div className="is-pop-title">{t("imgResolution")}</div>
                  <div className="is-bases">
                    <button type="button" className={settings.imgBase <= 0 ? "on" : ""} onClick={() => onSettings({ imgBase: 0 })}>
                      {t("imgAuto")} · {d.baseSize}
                    </button>
                    {BASE_SIZES.filter((b) => b !== d.baseSize).map((b) => (
                      <button key={b} type="button" className={settings.imgBase === b ? "on" : ""} onClick={() => onSettings({ imgBase: b })}>
                        {b}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </Pop>
          </div>

          <div className="is-bar-item is-stepper" title={t("imgBatch")}>
            <button disabled={eff.batch <= 1} onClick={() => onSettings({ imgBatch: eff.batch - 1 })}>−</button>
            <span>{t("imgCountN", { n: eff.batch })}</span>
            <button disabled={eff.batch >= 8} onClick={() => onSettings({ imgBatch: eff.batch + 1 })}>+</button>
          </div>

          <div className="is-bar-item">
            <button
              className={`is-bar-btn ${settings.imgSeedLock ? "on" : ""}`}
              title={settings.imgSeedLock ? t("imgSeedLocked") : t("imgSeedRandom")}
              onClick={() => onSettings({ imgSeedLock: !settings.imgSeedLock })}
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <rect x="4" y="4" width="16" height="16" rx="3" />
                <circle cx="9" cy="9" r="1" fill="currentColor" />
                <circle cx="15" cy="15" r="1" fill="currentColor" />
                {!settings.imgSeedLock && <circle cx="15" cy="9" r="1" fill="currentColor" />}
                {!settings.imgSeedLock && <circle cx="9" cy="15" r="1" fill="currentColor" />}
              </svg>
              {settings.imgSeedLock ? `${t("imgSeed")} ${settings.imgSeed}` : t("imgSeedRandom")}
            </button>
          </div>

          {d.negativePrompt && (
            <div className="is-bar-item">
              <button className={`is-bar-btn ${showNeg ? "on" : ""}`} onClick={() => setShowNeg((v) => !v)}>
                {t("imgNegative")}
              </button>
            </div>
          )}

          <div className="is-bar-item">
            <button className={`is-bar-btn ${pop === "params" ? "on" : ""}`} onClick={() => setPop(pop === "params" ? "" : "params")}>
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
                <circle cx="16" cy="7" r="2" />
                <circle cx="10" cy="17" r="2" />
              </svg>
              {t("imgStepsN", { n: eff.steps })} · CFG {eff.cfgScale}
            </button>
            <Pop open={pop === "params"} onClose={() => setPop("")} className="is-pop-params">
              <label className="is-field">
                <span>
                  {t("imgSteps")} <b>{eff.steps}</b>
                  {settings.imgSteps <= 0 && <em>{t("imgRecommended")}</em>}
                </span>
                <input type="range" min={1} max={100} step={1} value={eff.steps} onChange={(e) => onSettings({ imgSteps: Number(e.target.value) })} />
              </label>
              <label className="is-field">
                <span>
                  CFG <b>{eff.cfgScale}</b>
                  {settings.imgCfg <= 0 && <em>{t("imgRecommended")}</em>}
                </span>
                <input type="range" min={1} max={20} step={0.5} value={eff.cfgScale} onChange={(e) => onSettings({ imgCfg: Number(e.target.value) })} />
              </label>
              {eff.guidance != null && (
                <label className="is-field">
                  <span>
                    {t("imgGuidance")} <b>{eff.guidance}</b>
                  </span>
                  <input type="range" min={0} max={10} step={0.1} value={eff.guidance} onChange={(e) => onSettings({ imgGuidance: Number(e.target.value) })} />
                </label>
              )}
              <label className="is-field">
                <span>{t("imgSampler")}</span>
                <select value={settings.imgSampler} onChange={(e) => onSettings({ imgSampler: e.target.value })}>
                  <option value="">
                    {t("imgAuto")} ({d.sampler || info.defaultSampler || "—"})
                  </option>
                  {SAMPLERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="is-field">
                <span>{t("imgScheduler")}</span>
                <select value={settings.imgScheduler} onChange={(e) => onSettings({ imgScheduler: e.target.value })}>
                  <option value="">
                    {t("imgAuto")} ({d.scheduler || info.defaultScheduler || "—"})
                  </option>
                  {SCHEDULERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              {settings.imgSeedLock && (
                <label className="is-field">
                  <span>{t("imgSeed")}</span>
                  <span className="is-seed-row">
                    <input type="number" min={0} value={settings.imgSeed} onChange={(e) => onSettings({ imgSeed: Math.max(0, Number(e.target.value) || 0) })} />
                    <button
                      className="is-btn"
                      title={t("imgSeedDice")}
                      onClick={() => onSettings({ imgSeed: Math.floor(Math.random() * 4294967295) })}
                    >
                      🎲
                    </button>
                  </span>
                </label>
              )}
              <div className="is-pop-foot">
                <button
                  className="is-btn"
                  onClick={() => onSettings({ imgSteps: 0, imgCfg: 0, imgGuidance: 0, imgSampler: "", imgScheduler: "" })}
                >
                  {t("imgResetRecommended")}
                </button>
                <button
                  className="is-btn"
                  onClick={() => {
                    setPop("");
                    onOpenSettings();
                  }}
                >
                  {t("imgMoreSettings")}
                </button>
              </div>
            </Pop>
          </div>
        </div>
      </footer>
    </div>
  );
}
