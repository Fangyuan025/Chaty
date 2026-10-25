import { describe, expect, it } from "vitest";
import { classifyCatalogueFit, estimateStoreMemory } from "./storeMemory";

const MiB = 1024 * 1024;
const kind = { image: false, music: false, mlx: false };

describe("catalogue memory estimates", () => {
  it("does not invent dedicated VRAM for a Vulkan iGPU", () => {
    const info = estimateStoreMemory(1024 * MiB, 1024 * MiB, 5800, 16000, "Vulkan", "integrated");
    expect(info.unifiedMemory).toBe(true);
    expect(info.gpuBudgetBytes).toBeNull();
    expect(info.gpuFitsMainWeights).toBeNull();
    expect(info.gpuWeightsCeilingBytes).toBeNull();
    expect(classifyCatalogueFit(info, kind)).toBe("ram");
  });

  it("uses the selected discrete GPU budget, not total system RAM", () => {
    const info = estimateStoreMemory(1024 * MiB, 1024 * MiB, 4096, 16000, "Vulkan", "discrete");
    expect(info.unifiedMemory).toBe(false);
    expect(info.gpuBudgetBytes).toBe(4096 * MiB);
    expect(info.gpuFitsMainWeights).toBe(true);
    expect(classifyCatalogueFit(info, kind)).toBe("gpu");
  });

  it("does not trust virtual GPUs as dedicated hardware", () => {
    const info = estimateStoreMemory(1024 * MiB, 1024 * MiB, 4096, 16000, "Vulkan", "virtual");
    expect(info.gpuBudgetBytes).toBeNull();
    expect(classifyCatalogueFit(info, kind)).toBe("cpuOnly");
  });

  it("treats Metal as unified memory", () => {
    const info = estimateStoreMemory(1024 * MiB, 1024 * MiB, 10000, 16000, "Metal");
    expect(info.unifiedMemory).toBe(true);
    expect(info.gpuBudgetBytes).toBeNull();
    expect(classifyCatalogueFit(info, kind)).toBe("ram");
  });

  it("recognizes a GPU fit without applying a second memory margin", () => {
    const info = estimateStoreMemory(5500 * MiB, 6500 * MiB, 7000, 16000, "Vulkan", "discrete");
    expect(classifyCatalogueFit(info, kind)).toBe("gpu");
  });
  it("does not infer CPU offload from optional assets alone", () => {
    const info = estimateStoreMemory(5500 * MiB, 7500 * MiB, 7000, 16000, "Vulkan", "discrete");
    expect(classifyCatalogueFit(info, kind)).toBe("unknown");
  });
  it("reports a mixed fit when primary model weights exceed the budget", () => {
    const info = estimateStoreMemory(7500 * MiB, 7600 * MiB, 7000, 16000, "Vulkan", "discrete");
    expect(classifyCatalogueFit(info, kind)).toBe("mixed");
  });
  it("refuses a confident fit verdict when all budgets are unknown", () => {
    const info = estimateStoreMemory(1024 * MiB, 1024 * MiB);
    expect(classifyCatalogueFit(info, kind)).toBe("unknown");
  });
});
