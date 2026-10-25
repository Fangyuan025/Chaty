import { describe, expect, it } from "vitest";
import { classifyCatalogueFit, estimateStoreMemory, type CatalogueFit } from "./storeMemory";

const MiB = 1024 * 1024;
const GiB = 1024 * MiB;
const GB = 1_000_000_000;
const chat = { image: false, music: false, mlx: false };
const vision = chat; // Vision GGUF: projector is a supplementary download.
const image = { image: true, music: false, mlx: false };
const music = { image: false, music: true, mlx: false };
const mlx = { image: false, music: false, mlx: true };

type Kind = typeof chat;
type Case = {
  name: string;
  model: number;
  download: number;
  budget?: number | null;
  ram?: number | null;
  backend?: string | null;
  gpuKind?: "discrete" | "integrated" | "virtual" | "other" | null;
  capacity?: number | null;
  kind: Kind;
  expected: CatalogueFit;
};
const scenarios: Case[] = [
  // Size comparisons use raw bytes, including decimal GB displayed by the UI.
  { name: "chat weights fit despite optional downloads", model: 5.6 * GB, download: 6.5 * GB, budget: 5700, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "gpu" },
  { name: "vision projector does not become model weights", model: 6.7 * GB, download: 7.7 * GB, budget: 5500, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: vision, expected: "gpu" },
  { name: "image companions do not become model weights", model: 4.2 * GB, download: 10 * GB, budget: 5500, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: image, expected: "gpu" },
  { name: "music package overhead does not become model weights", model: 7.3 * GB, download: 7.5 * GB, budget: 5500, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: music, expected: "gpu" },
  { name: "very large chat model may need offload", model: 13 * GiB, download: 16 * GiB, budget: 5000, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "mixed" },
  { name: "large vision weights may need offload", model: 10 * GiB, download: 12 * GiB, budget: 5000, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: vision, expected: "mixed" },
  { name: "large image weights may need system RAM", model: 10 * GiB, download: 13 * GiB, budget: 5000, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: image, expected: "mixed" },
  { name: "large music weights may need CPU mode", model: 10 * GiB, download: 13 * GiB, budget: 5000, ram: 32000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: music, expected: "cpuMode" },
  { name: "model exceeds combined dedicated memory and RAM", model: 48 * GiB, download: 51 * GiB, budget: 5000, ram: 16000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "large" },
  { name: "CPU only with sufficient RAM", model: 5 * GiB, download: 8 * GiB, ram: 16000, backend: "CPU", kind: chat, expected: "cpuOnly" },
  { name: "CPU only with insufficient RAM", model: 24 * GiB, download: 25 * GiB, ram: 16000, backend: "CPU", kind: chat, expected: "large" },
  { name: "integrated GPU uses shared RAM", model: 6 * GiB, download: 9 * GiB, budget: 5500, ram: 16000, backend: "Vulkan", gpuKind: "integrated", kind: chat, expected: "ram" },
  { name: "integrated GPU does not add VRAM to RAM", model: 18 * GiB, download: 18 * GiB, budget: 5500, ram: 16000, backend: "Vulkan", gpuKind: "integrated", kind: chat, expected: "large" },
  { name: "Metal has unified memory", model: 7 * GiB, download: 8 * GiB, budget: 12000, ram: 16000, backend: "Metal", kind: chat, expected: "ram" },
  { name: "Metal model exceeds unified memory", model: 24 * GiB, download: 25 * GiB, budget: 12000, ram: 16000, backend: "Metal", kind: chat, expected: "large" },
  { name: "MLX uses RAM without a separate VRAM pool", model: 8 * GiB, download: 9 * GiB, budget: 12000, ram: 16000, backend: "Metal", kind: mlx, expected: "ram" },
  { name: "MLX oversized", model: 20 * GiB, download: 21 * GiB, budget: 12000, ram: 16000, backend: "Metal", kind: mlx, expected: "large" },
  { name: "virtual GPU is not dedicated VRAM", model: 3 * GiB, download: 3 * GiB, budget: 8192, ram: 16000, backend: "Vulkan", gpuKind: "virtual", kind: chat, expected: "cpuOnly" },
  { name: "other GPU class is not trusted", model: 3 * GiB, download: 4 * GiB, budget: 8192, ram: 16000, backend: "Vulkan", gpuKind: "other", kind: image, expected: "cpuOnly" },
  { name: "Windows-compatible IPC without kind", model: 4 * GiB, download: 6 * GiB, budget: 8192, ram: 16000, backend: "Vulkan", kind: chat, expected: "gpu" },
  { name: "positive physical VRAM survives missing dynamic budget", model: 7 * GiB, download: 8 * GiB, budget: null, ram: 16000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "gpu" },
  { name: "model exactly matches dedicated VRAM", model: 8192 * MiB, download: 8192 * MiB, budget: 2000, ram: 16000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "gpu" },
  { name: "model one byte beyond dedicated VRAM", model: 8192 * MiB + 1, download: 8192 * MiB + 1, budget: 2000, ram: 16000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "mixed" },
  { name: "small GPU with big RAM still allows partial offload", model: 6 * GiB, download: 10 * GiB, budget: 2000, ram: 24000, backend: "Vulkan", gpuKind: "discrete", capacity: 2048, kind: chat, expected: "mixed" },
  { name: "missing RAM prevents mixed-offload prediction", model: 12 * GiB, download: 13 * GiB, budget: 4000, ram: null, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "unknown" },
  { name: "missing all hardware information", model: 2 * GiB, download: 2 * GiB, kind: chat, expected: "unknown" },
  { name: "missing main file size", model: 0, download: 4 * GiB, budget: 5000, ram: 16000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "unknown" },
  { name: "negative main size is unknown", model: -1, download: 4 * GiB, budget: 5000, ram: 16000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "unknown" },
  { name: "nonfinite main size is unknown", model: Infinity, download: 4 * GiB, budget: 5000, ram: 16000, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: chat, expected: "unknown" },
  { name: "unknown GPU but verified RAM", model: 3 * GiB, download: 4 * GiB, ram: 16000, backend: "CPU", kind: music, expected: "cpuOnly" },
  { name: "GPU weight fit does not require RAM snapshot", model: 4 * GiB, download: 20 * GiB, budget: 2000, ram: null, backend: "Vulkan", gpuKind: "discrete", capacity: 8192, kind: image, expected: "gpu" },
];

describe("catalogue suitability across model families and memory topologies", () => {
  it.each(scenarios)("$name", ({ model, download, budget, ram, backend, gpuKind, capacity, kind, expected }) => {
    const estimate = estimateStoreMemory(model, download, budget, ram, backend, gpuKind, capacity);
    expect(classifyCatalogueFit(estimate, kind)).toBe(expected);
  });

  it("does not double-count integrated RAM or Metal memory", () => {
    for (const gpuKind of ["integrated"] as const) {
      const estimate = estimateStoreMemory(18 * GiB, 20 * GiB, 16000, 16000, "Vulkan", gpuKind, 8192);
      expect(estimate.gpuBudgetBytes).toBeNull();
      expect(estimate.gpuCapacityBytes).toBeNull();
      expect(estimate.unifiedMemory).toBe(true);
      expect(classifyCatalogueFit(estimate, chat)).toBe("large");
    }
  });

  it("separates physical capacity from fluctuating runtime Vulkan budget", () => {
    for (const budget of [null, 0, 512, 2500, 5700, 7000]) {
      const estimate = estimateStoreMemory(7 * GiB, 12 * GiB, budget, 32000, "Vulkan", "discrete", 8192);
      expect(estimate.gpuCapacityBytes).toBe(8192 * MiB);
      expect(estimate.gpuFitsMainWeights).toBe(true);
      expect(estimate.gpuBudgetBytes).toBe(budget && budget > 0 ? budget * MiB : null);
      expect(classifyCatalogueFit(estimate, chat)).toBe("gpu");
    }
  });

  it("never changes a fit verdict when only download extras change", () => {
    const variants: Kind[] = [chat, image, music, mlx];
    let comparisons = 0;
    for (const kind of variants) {
      for (const size of [1, 4, 7, 8, 12, 24, 42]) {
        const baseline = estimateStoreMemory(size * GiB, size * GiB, 5400, 32000, "Vulkan", "discrete", 8192);
        const reference = classifyCatalogueFit(baseline, kind);
        for (const extra of [0, 0.1, 2, 8, 24, 96]) {
          const estimate = estimateStoreMemory(size * GiB, (size + extra) * GiB, 5400, 32000, "Vulkan", "discrete", 8192);
          expect(classifyCatalogueFit(estimate, kind)).toBe(reference);
          comparisons++;
        }
      }
    }
    expect(comparisons).toBe(168);
  });

  it("never reports unknown when all essential hardware and model sizes are known", () => {
    const variants: Kind[] = [chat, image, music, mlx];
    let combinations = 0;
    for (const kind of variants) {
      for (const backend of ["Vulkan", "CPU", "Metal"]) {
        for (const gpuKind of ["discrete", "integrated", "virtual", "other"] as const) {
          for (const size of [1, 4, 8, 12, 24, 48]) {
            const estimate = estimateStoreMemory(size * GiB, 2 * size * GiB, 6000, 32000, backend, gpuKind, 8192);
            expect(classifyCatalogueFit(estimate, kind)).not.toBe("unknown");
            combinations++;
          }
        }
      }
    }
    expect(combinations).toBe(288);
  });

  it("does not mis-handle decimal GB versus binary GiB", () => {
    const capacity = 8192 * MiB;
    expect(8 * GB).toBeLessThan(capacity);
    expect(9 * GB).toBeGreaterThan(capacity);
    expect(classifyCatalogueFit(estimateStoreMemory(8 * GB, 10 * GB, 6000, 32000, "Vulkan", "discrete", 8192), chat)).toBe("gpu");
    expect(classifyCatalogueFit(estimateStoreMemory(9 * GB, 10 * GB, 6000, 32000, "Vulkan", "discrete", 8192), chat)).toBe("mixed");
  });

  it("does not invent a memory fit from invalid numbers", () => {
    const invalid = estimateStoreMemory(Number.NaN, Number.POSITIVE_INFINITY, -10, Number.NaN, "Vulkan", "discrete", -100);
    expect(invalid.modelBytes).toBe(0);
    expect(invalid.gpuBudgetBytes).toBeNull();
    expect(invalid.gpuCapacityBytes).toBeNull();
    expect(classifyCatalogueFit(invalid, chat)).toBe("unknown");
  });

  it("clamps inconsistent download metadata without altering main model size", () => {
    const estimate = estimateStoreMemory(5 * GiB, 2 * GiB, 5500, 16000, "Vulkan", "discrete", 8192);
    expect(estimate.modelBytes).toBe(5 * GiB);
    expect(estimate.downloadBytes).toBe(5 * GiB);
    expect(estimate.extraBytes).toBe(0);
    expect(classifyCatalogueFit(estimate, chat)).toBe("gpu");
  });

  it("cannot produce a worse GPU fit by increasing only physical GPU capacity", () => {
    const rank: Record<CatalogueFit, number> = { large: 0, unknown: 0, cpuMode: 1, mixed: 1, cpuOnly: 1, ram: 1, gpu: 2 };
    for (const kind of [chat, image, music]) {
      for (const size of [1, 4, 7, 10, 14, 30, 48]) {
        let previous = -1;
        for (const capacity of [1024, 4096, 8192, 16384, 32768]) {
          const estimate = estimateStoreMemory(size * GiB, (size + 3) * GiB, 500, 32000, "Vulkan", "discrete", capacity);
          const current = rank[classifyCatalogueFit(estimate, kind)];
          expect(current >= previous).toBe(true);
          previous = current;
        }
      }
    }
  });
});
