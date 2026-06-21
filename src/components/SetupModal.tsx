import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";
import {
  cancelDownload,
  downloadModel,
  DOWNLOAD_CANCELLED,
  getHardwareInfo,
  listHfGgufs,
  type DownloadProgress,
} from "../lib/ipc";

/** One hardware-fitted recommendation (a model family at a size + quant). */
interface Pick {
  family: string;
  label: string;
  quant: string;
  approxGb: number;
  /** Candidate HF repos, tried in order until one lists GGUFs. */
  repos: string[];
  blurbZh: string;
  blurbEn: string;
}

/**
 * Chaty's own fine-tune — offered in every tier (small + specialised for the
 * on-device web-design workflow, with baked-in Chaty identity and grounding).
 */
const CHATY_PICK: Pick = {
  family: "Chaty",
  label: "Chaty Design 4B",
  quant: "Q4_K_M",
  approxGb: 2.7,
  repos: ["stevenpr/chaty-qwen3.5-4b-design-GGUF"],
  blurbZh:
    "Chaty 专属微调（基于 Qwen3.5-4B）：单文件网页设计更强、输出更精简，并内置 Chaty 身份与引用规范，轻量设备也跑得动。",
  blurbEn:
    "Chaty's own fine-tune (Qwen3.5-4B): stronger, leaner single-file web design, with built-in Chaty identity & grounded citations — runs on light machines.",
};

/**
 * Curated picks per memory budget. Sizes are the Q4_K_M weights; the budget
 * additionally needs room for the KV cache, so tiers are conservative.
 * Every tier also offers the Chaty fine-tune as a specialised option.
 */
function recommend(budgetGb: number): Pick[] {
  if (budgetGb >= 30) {
    return [
      {
        family: "Qwen3.5",
        label: "Qwen3.5 35B-A3B",
        quant: "Q4_K_M",
        approxGb: 21,
        repos: [
          "unsloth/Qwen3.5-35B-A3B-GGUF",
          "AesSedai/Qwen3.5-35B-A3B-GGUF",
          "Qwen/Qwen3.5-35B-A3B-GGUF",
        ],
        blurbZh: "旗舰级 MoE：35B 总参数、每次只激活 3B，速度快、能力强，本机内存足以全 GPU 运行。",
        blurbEn: "Flagship MoE: 35B total / 3B active — fast and capable, fits fully on your GPU memory.",
      },
      {
        family: "Gemma 4",
        label: "Gemma 4 26B-A4B",
        quant: "Q4_K_M",
        approxGb: 16,
        repos: [
          "unsloth/gemma-4-26B-A4B-it-GGUF",
          "unsloth/gemma-4-26b-a4b-it-GGUF",
          "bartowski/google_gemma-4-26B-A4B-it-GGUF",
        ],
        blurbZh: "Google 最新 MoE：综合素质均衡，思考模式表现好，多语言能力突出。",
        blurbEn: "Google's latest MoE: balanced quality, strong thinking mode and multilingual skills.",
      },
      CHATY_PICK,
    ];
  }
  if (budgetGb >= 12) {
    return [
      {
        family: "Qwen3.5",
        label: "Qwen3.5 9B",
        quant: "Q5_K_M",
        approxGb: 6.5,
        repos: ["unsloth/Qwen3.5-9B-GGUF", "Qwen/Qwen3.5-9B-GGUF"],
        blurbZh: "9B 稠密模型的高保真量化：日常对话与代码能力俱佳，仍留有充足上下文内存。",
        blurbEn: "Dense 9B at a high-fidelity quant — great chat & code, with memory to spare for context.",
      },
      {
        family: "Gemma 4",
        label: "Gemma 4 E4B",
        quant: "Q5_K_M",
        approxGb: 6,
        repos: ["unsloth/gemma-4-E4B-it-GGUF", "bartowski/google_gemma-4-E4B-it-GGUF"],
        blurbZh: "MatFormer 架构的高效 8B：有效 4B 计算量，速度与质量的甜点位。",
        blurbEn: "Efficient MatFormer 8B (≈4B effective compute) — a sweet spot of speed and quality.",
      },
      CHATY_PICK,
    ];
  }
  if (budgetGb >= 7) {
    return [
      {
        family: "Qwen3.5",
        label: "Qwen3.5 9B",
        quant: "Q4_K_M",
        approxGb: 5.5,
        repos: ["unsloth/Qwen3.5-9B-GGUF", "Qwen/Qwen3.5-9B-GGUF"],
        blurbZh: "9B 的标准量化：能力与内存占用的均衡之选。",
        blurbEn: "Standard quant of the dense 9B — balanced quality vs memory.",
      },
      {
        family: "Gemma 4",
        label: "Gemma 4 E4B",
        quant: "Q4_K_M",
        approxGb: 5,
        repos: ["unsloth/gemma-4-E4B-it-GGUF", "bartowski/google_gemma-4-E4B-it-GGUF"],
        blurbZh: "高效 8B（有效 4B），在这档内存上运行流畅。",
        blurbEn: "Efficient 8B (≈4B effective) that runs comfortably in this memory class.",
      },
      CHATY_PICK,
    ];
  }
  return [
    {
      family: "Qwen3.5",
      label: "Qwen3.5 4B",
      quant: "Q4_K_M",
      approxGb: 2.7,
      repos: ["unsloth/Qwen3.5-4B-GGUF", "Qwen/Qwen3.5-4B-GGUF"],
      blurbZh: "小而能打的 4B：轻量设备的最佳起点。",
      blurbEn: "Small but capable 4B — the best starting point for lighter machines.",
    },
    {
      family: "Gemma 4",
      label: "Gemma 4 E2B",
      quant: "Q4_K_M",
      approxGb: 3,
      repos: ["unsloth/gemma-4-E2B-it-GGUF", "bartowski/google_gemma-4-E2B-it-GGUF"],
      blurbZh: "Gemma 4 最小档：响应快、占用低。",
      blurbEn: "The smallest Gemma 4 — quick responses, low footprint.",
    },
    CHATY_PICK,
  ];
}

/** Preference order when matching a quant inside a repo's file list. */
const QUANT_FALLBACK = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0"];

type CardState =
  | { kind: "idle" }
  | { kind: "resolving" }
  | { kind: "downloading"; pct: number; file: string }
  | { kind: "done"; path: string }
  | { kind: "error"; message: string };

export function SetupModal({
  onClose,
  onLoad,
}: {
  onClose: () => void;
  /** Load the freshly downloaded model (path) and close. */
  onLoad: (path: string) => void;
}) {
  const { t, lang } = useI18n();
  const [budgetGb, setBudgetGb] = useState<number | null>(null);
  const [hwLine, setHwLine] = useState("");
  const [states, setStates] = useState<Record<string, CardState>>({});

  useEffect(() => {
    getHardwareInfo()
      .then((hw) => {
        // Unified memory (macOS) reports the Metal working set as "VRAM";
        // discrete GPUs report real VRAM; CPU-only machines get half the RAM.
        const gb = (hw.gpu?.vramMb ?? hw.ramMb / 2) / 1024;
        setBudgetGb(gb);
        setHwLine(
          `${hw.gpu?.name ?? hw.cpu} · ${(hw.ramMb / 1024).toFixed(0)} GB RAM · ${t(
            "setupBudget",
          )} ${gb.toFixed(0)} GB`,
        );
      })
      .catch(() => setBudgetGb(8));
  }, [t]);

  const setCard = (key: string, s: CardState) =>
    setStates((prev) => ({ ...prev, [key]: s }));

  async function download(p: Pick) {
    setCard(p.label, { kind: "resolving" });
    try {
      // Resolve: first candidate repo that lists GGUFs wins.
      let chosen: { name: string; url: string } | null = null;
      let lastErr = "";
      for (const repo of p.repos) {
        try {
          const files = (await listHfGgufs(repo)).filter(
            (f) => !/-\d{5}-of-/.test(f.name), // skip multi-part shards
          );
          for (const q of [p.quant, ...QUANT_FALLBACK]) {
            const hit = files.find((f) => f.name.toUpperCase().includes(q));
            if (hit) {
              chosen = hit;
              break;
            }
          }
          if (chosen) break;
        } catch (e) {
          lastErr = e instanceof Error ? e.message : String(e);
        }
      }
      if (!chosen) {
        setCard(p.label, {
          kind: "error",
          message: lastErr || t("setupNotFound"),
        });
        return;
      }
      const file = chosen.name;
      setCard(p.label, { kind: "downloading", pct: 0, file });
      await downloadModel(chosen.url, file, (ev: DownloadProgress) => {
        if (ev.type === "progress" && ev.total > 0) {
          setCard(p.label, {
            kind: "downloading",
            pct: Math.round((ev.downloaded / ev.total) * 100),
            file,
          });
        } else if (ev.type === "done") {
          setCard(p.label, { kind: "done", path: ev.path });
        } else if (ev.type === "error" && ev.message !== DOWNLOAD_CANCELLED) {
          setCard(p.label, { kind: "error", message: ev.message });
        }
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setCard(
        p.label,
        message === DOWNLOAD_CANCELLED
          ? { kind: "idle" }
          : { kind: "error", message },
      );
    }
  }

  const picks = budgetGb === null ? [] : recommend(budgetGb);

  return createPortal(
    <div className="preview-overlay" onMouseDown={onClose}>
      <div className="setup-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="setup-head">
          <div>
            <div className="setup-title">{t("setupTitle")}</div>
            <div className="setup-hw">{hwLine || "…"}</div>
          </div>
          <button className="preview-close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="setup-cards">
          {picks.map((p) => {
            const st = states[p.label] ?? { kind: "idle" as const };
            return (
              <div key={p.label} className="setup-card">
                <div className="setup-card-family">{p.family}</div>
                <div className="setup-card-name">{p.label}</div>
                <div className="setup-card-meta">
                  {p.quant} · ≈{p.approxGb} GB
                </div>
                <div className="setup-card-blurb">
                  {lang === "zh" ? p.blurbZh : p.blurbEn}
                </div>
                {st.kind === "idle" || st.kind === "error" ? (
                  <>
                    <button className="setup-dl" onClick={() => void download(p)}>
                      {t("setupDownload")}
                    </button>
                    {st.kind === "error" && (
                      <div className="setup-err">{st.message.slice(0, 160)}</div>
                    )}
                  </>
                ) : st.kind === "resolving" ? (
                  <button className="setup-dl" disabled>
                    {t("setupResolving")}
                  </button>
                ) : st.kind === "downloading" ? (
                  <div className="setup-progress-row">
                    <div className="setup-progress">
                      <div className="setup-progress-fill" style={{ width: `${st.pct}%` }} />
                      <span>{st.pct}%</span>
                    </div>
                    <button
                      className="dl-cancel"
                      title={t("cancel")}
                      onClick={() => void cancelDownload(st.file).catch(() => {})}
                    >
                      ×
                    </button>
                  </div>
                ) : (
                  <button className="setup-dl ready" onClick={() => onLoad(st.path)}>
                    {t("setupLoad")}
                  </button>
                )}
              </div>
            );
          })}
        </div>
        <div className="setup-foot">{t("setupFoot")}</div>
      </div>
    </div>,
    document.body,
  );
}
