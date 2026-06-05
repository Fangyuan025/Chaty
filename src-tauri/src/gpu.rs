//! GPU detection + auto‑tuning of how many model layers to offload.
//!
//! Detection uses DXGI (cross‑vendor on Windows: NVIDIA / AMD / Intel). The
//! auto‑tuner estimates how many transformer layers fit in VRAM and offloads as
//! many as it can, leaving the rest on the CPU. `llama.rs` additionally retries
//! with fewer layers if a GPU allocation fails, so an over‑estimate is safe.

use std::path::Path;

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub cpu: String,
    pub cpu_threads: usize,
    pub ram_mb: u64,
    pub gpu: Option<GpuInfo>,
    /// Compiled GPU backend for the LLM: "Vulkan" when built with the `gpu`
    /// feature, otherwise "CPU".
    pub gpu_backend: String,
}

/// Snapshot of the machine's CPU / RAM / GPU and the compiled GPU backend.
pub fn hardware() -> HardwareInfo {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let cpu = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    HardwareInfo {
        cpu,
        cpu_threads,
        ram_mb: sys.total_memory() / (1024 * 1024),
        gpu: detect_gpu(),
        gpu_backend: if cfg!(feature = "gpu") {
            "Vulkan".to_string()
        } else {
            "CPU".to_string()
        },
    }
}

/// The discrete GPU with the most VRAM, or `None` if there's no usable GPU.
#[cfg(windows)]
pub fn detect_gpu() -> Option<GpuInfo> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
    };

    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<GpuInfo> = None;
        let mut i = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            i += 1;
            let desc = match adapter.GetDesc1() {
                Ok(d) => d,
                Err(_) => continue,
            };
            // Skip the software / WARP renderer.
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }
            let vram_mb = (desc.DedicatedVideoMemory as u64) / (1024 * 1024);
            let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..end])
                .trim()
                .to_string();
            if best.as_ref().map_or(true, |b| vram_mb > b.vram_mb) {
                best = Some(GpuInfo { name, vram_mb });
            }
        }
        best.filter(|g| g.vram_mb > 0)
    }
}

#[cfg(not(windows))]
pub fn detect_gpu() -> Option<GpuInfo> {
    None
}

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuUsage {
    /// VRAM currently in use by all apps, in MB.
    pub used_mb: u64,
    /// Total dedicated VRAM, in MB.
    pub total_mb: u64,
}

/// Live VRAM usage of the primary GPU (DXGI 1.4 `QueryVideoMemoryInfo`).
#[cfg(windows)]
pub fn gpu_usage() -> Option<GpuUsage> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };

    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<(u64, IDXGIAdapter3)> = None;
        let mut i = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            i += 1;
            let desc = match adapter.GetDesc1() {
                Ok(d) => d,
                Err(_) => continue,
            };
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }
            let vram = desc.DedicatedVideoMemory as u64;
            if let Ok(a3) = adapter.cast::<IDXGIAdapter3>() {
                if best.as_ref().map_or(true, |(v, _)| vram > *v) {
                    best = Some((vram, a3));
                }
            }
        }
        let (total, adapter) = best?;
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
            .ok()?;
        Some(GpuUsage {
            used_mb: info.CurrentUsage / (1024 * 1024),
            total_mb: total / (1024 * 1024),
        })
    }
}

#[cfg(not(windows))]
pub fn gpu_usage() -> Option<GpuUsage> {
    None
}

/// How many layers (incl. the output layer) to offload to fill VRAM.
///
/// `n_layer` is the model's transformer block count; `vram_mb` the GPU's total
/// dedicated memory. We reserve headroom for the KV cache, compute buffers and
/// whatever the desktop/other apps already use, then offload as many whole
/// layers as the remaining budget holds.
pub fn auto_gpu_layers(path: &Path, n_layer: u32, vram_mb: u64) -> i32 {
    if vram_mb == 0 || n_layer == 0 {
        return 0;
    }
    let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if file_bytes == 0 {
        return 0;
    }
    // Weights are roughly evenly spread across blocks (+1 for embeddings/output).
    let per_layer = file_bytes as f64 / (n_layer as f64 + 1.0);

    let vram = vram_mb as f64 * 1024.0 * 1024.0;
    // Reserve ~20% or at least 1 GiB for the KV cache / compute buffers / OS.
    let reserve = (vram * 0.20).max(1024.0 * 1024.0 * 1024.0);
    let budget = (vram - reserve).max(0.0);

    let fit = (budget / per_layer).floor() as i64;
    fit.clamp(0, n_layer as i64 + 1) as i32
}
