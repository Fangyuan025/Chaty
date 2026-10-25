/** Catalogue-only memory hint, never a runtime allocator.
 * File download size is not GPU residency. In particular, extra VAE,
 * projector, and audio assets can be placed or loaded separately.
 * The catalogue compares primary weights with device capacity; the runtime
 * uses its own currently available Vulkan budget, context, and buffers.
 */
export interface StoreMemoryEstimate {
  modelBytes: number;
  extraBytes: number;
  downloadBytes: number;
  ramBytes: number | null;
  /** Runtime allocation budget, when the backend exposes one. */
  gpuBudgetBytes: number | null;
  /** Capacity used for a stable pre-download fit hint. */
  gpuCapacityBytes: number | null;
  /** Primary model weights only, NOT a promise of runtime GPU residency. */
  gpuFitsMainWeights: boolean | null;
  /** Budget-based, theoretical weight bounds, not a layer count. */
  gpuWeightsCeilingBytes: number | null;
  cpuWeightsFloorBytes: number | null;
  modelOverInstalledRam: boolean | null;
  unifiedMemory: boolean;
}

const MIB = 1024 * 1024;
const positive = (n?: number | null): n is number =>
  n != null && Number.isFinite(n) && n > 0;

/** IPC memory quantities are MiB; Hugging Face file sizes are bytes. */
export function estimateStoreMemory(
  modelBytes: number,
  downloadBytes: number,
  vramMb?: number | null,
  ramMb?: number | null,
  gpuBackend?: string | null,
  gpuKind?: "discrete" | "integrated" | "virtual" | "other" | null,
  dedicatedVramMb?: number | null,
): StoreMemoryEstimate {
  const model = positive(modelBytes) ? modelBytes : 0;
  const download = positive(downloadBytes) ? Math.max(model, downloadBytes) : model;
  const ramBytes = positive(ramMb) ? ramMb * MIB : null;

  // UMA/Metal uses system RAM; never add it to a fictitious VRAM pool.
  const unifiedMemory = gpuBackend === "Metal" || gpuKind === "integrated";
  const usableSeparateGpu = gpuBackend !== "CPU" && !unifiedMemory &&
    gpuKind !== "virtual" && gpuKind !== "other";

  const gpuBudgetBytes = usableSeparateGpu && positive(vramMb)
    ? vramMb * MIB : null;
  // Linux provides physical VRAM separately from its fluctuating Vulkan
  // allocation budget. On other platforms, retain the existing IPC fallback.
  const gpuCapacityBytes = !usableSeparateGpu ? null
    : gpuKind === "discrete" && positive(dedicatedVramMb)
      ? dedicatedVramMb * MIB
      : gpuBudgetBytes;

  // These optional diagnostics continue to use the *runtime* budget, not
  // the catalogue capacity. They must not be summed with unified RAM.
  const planned = gpuBudgetBytes !== null && model > 0;
  const gpuWeightsCeilingBytes = planned ? Math.min(model, gpuBudgetBytes) : null;
  const cpuWeightsFloorBytes = planned
    ? Math.max(0, model - gpuWeightsCeilingBytes!) : null;

  return {
    modelBytes: model,
    extraBytes: download - model,
    downloadBytes: download,
    ramBytes,
    gpuBudgetBytes,
    gpuCapacityBytes,
    gpuFitsMainWeights: gpuCapacityBytes !== null && model > 0
      ? model <= gpuCapacityBytes : null,
    gpuWeightsCeilingBytes,
    cpuWeightsFloorBytes,
    modelOverInstalledRam: ramBytes !== null && model > 0
      ? model > ramBytes : null,
    unifiedMemory,
  };
}

export type CatalogueFit =
  | "gpu"
  | "mixed"
  | "cpuMode"
  | "cpuOnly"
  | "ram"
  | "large"
  | "unknown";

/** File-size-only suitability hint, never equivalent to LM Studio's
 * model-aware estimator (context, KV cache, architecture, and buffers).
 * Return "unknown" only when essential data is genuinely missing.
 */
export function classifyCatalogueFit(
  estimate: StoreMemoryEstimate | null,
  kind: { image: boolean; music: boolean; mlx: boolean },
): CatalogueFit {
  if (!estimate || !positive(estimate.modelBytes)) return "unknown";
  const model = estimate.modelBytes;
  const ram = estimate.ramBytes;
  const gpu = estimate.gpuCapacityBytes;

  if (estimate.unifiedMemory || kind.mlx) {
    if (ram === null) return "unknown";
    return model <= ram ? "ram" : "large";
  }

  if (gpu !== null) {
    // The badge describes primary model weights. Supplementary downloads do
    // not necessarily occupy GPU memory at the same time as the main model.
    if (model <= gpu) return "gpu";
    if (ram === null) return "unknown";
    // CPU RAM and dedicated VRAM are separate pools only in this branch.
    if (model > gpu + ram) return "large";
    return kind.music ? "cpuMode" : "mixed";
  }

  if (ram === null) return "unknown";
  return model <= ram ? "cpuOnly" : "large";
}
