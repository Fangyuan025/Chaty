/** Catalogue-only preview, not an inference memory allocator.
 *
 * A download's byte count is NOT peak RAM/VRAM. GPU reporting differs by
 * platform: Chaty's Linux Vulkan backend reports a conservative selected-GPU
 * allocation budget; Windows may report total dedicated VRAM; Metal has
 * unified memory. Never promise that a model will load based on these sizes.
 */
export interface StoreMemoryEstimate {
  modelBytes: number;
  extraBytes: number;
  downloadBytes: number;
  ramBytes: number | null;
  gpuBudgetBytes: number | null;
  /** null = no separate dedicated-GPU estimate (or unknown). */
  gpuFitsMainWeights: boolean | null;
  /** A *theoretical upper bound* for llama.cpp model weights, NOT layer count. */
  gpuWeightsCeilingBytes: number | null;
  /** A *theoretical lower bound* for CPU-resident model weights. */
  cpuWeightsFloorBytes: number | null;
  modelOverInstalledRam: boolean | null;
  unifiedMemory: boolean;
}

/** Values in the IPC hardware snapshot are MiB, but file sizes are bytes. */
export function estimateStoreMemory(
  modelBytes: number,
  downloadBytes: number,
  vramMb?: number | null,
  ramMb?: number | null,
  gpuBackend?: string | null,
  gpuKind?: "discrete" | "integrated" | "virtual" | "other" | null,
): StoreMemoryEstimate {
  const model = Number.isFinite(modelBytes) ? Math.max(0, modelBytes) : 0;
  const download = Number.isFinite(downloadBytes) ? Math.max(model, downloadBytes) : model;
  const mib = 1024 * 1024;
  const ramBytes = ramMb != null && Number.isFinite(ramMb) && ramMb > 0 ? ramMb * mib : null;
  // Integrated Vulkan GPUs and Metal share system RAM.
  const unifiedMemory =
    gpuBackend === "Metal" || gpuKind === "integrated";
  const gpuBudgetBytes = !unifiedMemory && gpuKind !== "virtual" &&
    gpuKind !== "other" &&
    gpuBackend !== "CPU" && vramMb != null && Number.isFinite(vramMb) && vramMb > 0
    ? vramMb * mib
    : null;
  // Metal's budget and system RAM describe the SAME shared pool, not two
  // independent memory pools, so it is misleading to calculate GPU/CPU split.
  const separateBudget = !unifiedMemory && gpuBudgetBytes != null && model > 0;
  const gpuWeightsCeilingBytes = separateBudget ? Math.min(model, gpuBudgetBytes) : null;
  const cpuWeightsFloorBytes = separateBudget ? Math.max(0, model - gpuWeightsCeilingBytes!) : null;
  return {
    modelBytes: model,
    extraBytes: download - model,
    downloadBytes: download,
    ramBytes,
    gpuBudgetBytes,
    gpuFitsMainWeights: separateBudget ? model <= gpuBudgetBytes : null,
    gpuWeightsCeilingBytes,
    cpuWeightsFloorBytes,
    modelOverInstalledRam: ramBytes != null && model > 0 ? model > ramBytes : null,
    unifiedMemory,
  };
}

/** A short, deliberately conservative hint for the model catalogue.
 * The backend only reports installed RAM and a GPU budget. It cannot predict
 * runtime memory precisely, so this is never a guarantee or a layer count.
 */
export type CatalogueFit =
  | "gpu"
  | "mixed"
  | "cpuMode"
  | "cpuOnly"
  | "ram"
  | "large"
  | "unknown";

export function classifyCatalogueFit(
  estimate: StoreMemoryEstimate | null,
  kind: { image: boolean; music: boolean; mlx: boolean },
): CatalogueFit {
  if (!estimate || estimate.downloadBytes <= 0) return "unknown";
  const needed = estimate.downloadBytes;
  const ram = estimate.ramBytes;
  const gpu = estimate.gpuBudgetBytes;

  // Metal shares RAM with its GPU. Never add a second, fictitious VRAM pool.
  if (estimate.unifiedMemory || kind.mlx) {
    if (ram === null) return "unknown";
    if (needed <= ram * 0.7) return "ram";
    return needed > ram * 0.85 ? "large" : "unknown";
  }

  // The Linux Vulkan allocation budget already includes a safety margin.
  // Do not subtract another arbitrary margin for this catalogue hint.
  // File sizes are proxies; a positive result is not a loading guarantee.
  if (gpu !== null) {
    if (needed <= gpu) return "gpu";
    // Optional assets may not reside on the GPU simultaneously. If the main
    // model weights fit, we cannot infer that CPU offload is required.
    if (estimate.gpuFitsMainWeights === true) return "unknown";
    const systemRemainder = Math.max(0, estimate.modelBytes - gpu);
    if (ram !== null && systemRemainder > ram * 0.75) return "large";
    if (kind.music) return "cpuMode";
    return "mixed";
  }

  if (ram !== null) {
    if (needed <= ram * 0.7) return "cpuOnly";
    if (needed > ram * 0.85) return "large";
  }
  return "unknown";
}
