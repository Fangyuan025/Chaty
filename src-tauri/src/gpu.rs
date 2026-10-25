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
    /// Linux/Metal: offload budget; Windows: reported dedicated capacity.
    pub vram_mb: u64,
    /// Physical dedicated VRAM in MiB, when separately measurable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dedicated_vram_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub cpu: String,
    pub cpu_threads: usize,
    pub ram_mb: u64,
    pub gpu: Option<GpuInfo>,
    /// Compiled GPU backend for the LLM: "Metal" on macOS, "Vulkan" on
    /// Windows/Linux (both when built with the `gpu` feature), otherwise "CPU".
    pub gpu_backend: String,
}

/// The compiled GPU backend name, gated on the `gpu` feature and the target.
fn gpu_backend_name() -> String {
    if cfg!(feature = "gpu") {
        if cfg!(target_os = "macos") {
            "Metal".to_string()
        } else {
            "Vulkan".to_string()
        }
    } else {
        "CPU".to_string()
    }
}

/// Number of worker threads for CPU-side work (tokenizer, sampling, any
/// non-offloaded layers). On Apple Silicon (big.LITTLE) we use only the
/// performance cores and leave one free for the UI; spawning onto efficiency
/// cores hurts throughput. Elsewhere we use the logical CPU count.
pub fn cpu_worker_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(p) = perf_core_count() {
            return p.saturating_sub(1).max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Performance-core count on Apple Silicon via `hw.perflevel0.physicalcpu`.
#[cfg(target_os = "macos")]
fn perf_core_count() -> Option<usize> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.physicalcpu"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse::<usize>().ok()
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
        gpu_backend: gpu_backend_name(),
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
                best = Some(GpuInfo { name, vram_mb, dedicated_vram_mb: None, kind: None });
            }
        }
        best.filter(|g| g.vram_mb > 0)
    }
}

/// macOS / Apple Silicon: there is no discrete VRAM. We report the Metal
/// **recommended max working-set size** as the offload budget (the safe cap for
/// GPU allocations) and the Metal device name (e.g. "Apple M3 Pro") as the GPU.
#[cfg(target_os = "macos")]
pub fn detect_gpu() -> Option<GpuInfo> {
    let device = metal::Device::system_default()?;
    let working_set = device.recommended_max_working_set_size();
    let name = {
        let n = device.name().trim().to_string();
        if n.is_empty() { mac_chip_name() } else { n }
    };
    Some(GpuInfo {
        name,
        vram_mb: working_set / (1024 * 1024),
        dedicated_vram_mb: None,
        kind: None,
    })
}

/// Chip brand string fallback (e.g. "Apple M3 Pro") via sysctl.
#[cfg(target_os = "macos")]
fn mac_chip_name() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Apple GPU".to_string())
}

// Linux: query the Vulkan loader directly. Do not depend on vulkaninfo or
// PCI vendor-specific tools. Keep all devices for the later selection UI.
#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
pub struct LinuxGpu {
    pub index: usize,
    pub name: String,
    pub kind: &'static str,
    pub vendor_id: u32,
    pub device_id: u32,
    pub memory_mb: u64,
    pub budget_mb: u64,
}

// Returns a *remaining allocation budget* for this process, not total VRAM.
// VK_EXT_memory_budget accounts for memory pressure from other applications in
// heapBudget and for allocations by this process in heapUsage. The extension
// is optional; Vulkan 1.0 devices take the conservative fallback below.
#[cfg(target_os = "linux")]
fn linux_vulkan_free_mb(
    instance: &ash::Instance,
    device: ash::vk::PhysicalDevice,
    memory: &ash::vk::PhysicalDeviceMemoryProperties,
    supports_properties2: bool,
) -> Option<u64> {
    use ash::vk;
    use std::ffi::CStr;

    if !supports_properties2 {
        return None;
    }
    let props = unsafe { instance.get_physical_device_properties(device) };
    if props.api_version < vk::API_VERSION_1_1 {
        return None;
    }
    let extensions = unsafe { instance.enumerate_device_extension_properties(device) }.ok()?;
    let supported = extensions.iter().any(|ext| {
        unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }.to_bytes()
            == b"VK_EXT_memory_budget"
    });
    if !supported {
        return None;
    }

    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    {
        let mut props2 = vk::PhysicalDeviceMemoryProperties2::default()
            .push_next(&mut budget);
        unsafe { instance.get_physical_device_memory_properties2(device, &mut props2) };
    }
    let mut total_free = 0_u64;
    let mut found_local_heap = false;
    for (i, heap) in memory.memory_heaps[..memory.memory_heap_count as usize]
        .iter().enumerate()
    {
        if !heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            continue;
        }
        let max_for_process = budget.heap_budget[i].min(heap.size);
        if max_for_process == 0 {
            continue;
        }
        found_local_heap = true;
        total_free = total_free.saturating_add(
            max_for_process.saturating_sub(budget.heap_usage[i]),
        );
    }
    found_local_heap.then_some(total_free / (1024 * 1024))
}

// Conservative policy shared by AMD, Intel, NVIDIA and other Vulkan drivers.
// An iGPU shares system RAM: never plan offload from the entire local heap.
// A dGPU without VK_EXT_memory_budget keeps a safety margin; fit_layers()
// separately reserves memory for KV cache and compute buffers.
#[cfg(target_os = "linux")]
fn linux_offload_budget_mb(
    kind: &str,
    memory_mb: u64,
    available_ram_mb: u64,
    driver_free_mb: Option<u64>,
) -> u64 {
    let fallback = if kind == "integrated" {
        memory_mb.min(available_ram_mb / 4)
    } else {
        memory_mb.saturating_mul(85) / 100
    };
    driver_free_mb.map_or(fallback, |free| fallback.min(free))
}

#[cfg(all(test, target_os = "linux"))]
mod linux_budget_tests {
    use super::linux_offload_budget_mb;

    #[test]
    fn integrated_budget_is_capped_by_available_system_ram() {
        assert_eq!(linux_offload_budget_mb("integrated", 32_768, 8_000, None), 2_000);
    }

    #[test]
    fn discrete_budget_respects_driver_free_memory() {
        assert_eq!(linux_offload_budget_mb("discrete", 8_192, 16_000, Some(1_500)), 1_500);
    }

    #[test]
    fn missing_driver_budget_uses_safe_fallback() {
        assert_eq!(linux_offload_budget_mb("discrete", 8_000, 16_000, None), 6_800);
    }
}

#[cfg(target_os = "linux")]
pub fn linux_vulkan_gpus() -> Vec<LinuxGpu> {
    use ash::vk;
    use std::ffi::CStr;

    // A missing/broken loader must not prevent a CPU-only launch.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    // Try Vulkan 1.1 to query memory budgets. Vulkan 1.0 still enumerates
    // GPUs and falls back to a static, conservative budget.
    let loader_api = unsafe { entry.try_enumerate_instance_version() }
        .ok().flatten().unwrap_or(vk::API_VERSION_1_0);
    let mut instance = None;
    let mut supports_properties2 = false;
    for api in [vk::API_VERSION_1_1, vk::API_VERSION_1_0] {
        if api > loader_api {
            continue;
        }
        let app_info = vk::ApplicationInfo::default().api_version(api);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        if let Ok(i) = unsafe { entry.create_instance(&create_info, None) } {
            supports_properties2 = api >= vk::API_VERSION_1_1;
            instance = Some(i);
            break;
        }
    }
    let instance = match instance {
        Some(i) => i,
        None => return Vec::new(),
    };
    let handles = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available_ram_mb = sys.available_memory() / (1024 * 1024);
    let mut result = Vec::new();

    for (index, device) in handles.into_iter().enumerate() {
        let props = unsafe { instance.get_physical_device_properties(device) };
        let kind = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
            vk::PhysicalDeviceType::CPU => continue,
            _ => "other",
        };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let lower = name.to_ascii_lowercase();
        if ["llvmpipe", "lavapipe", "softpipe", "swiftshader"]
            .iter()
            .any(|tag| lower.contains(tag))
        {
            continue;
        }
        // llama.cpp requires a compute-capable Vulkan queue.
        if !unsafe { instance.get_physical_device_queue_family_properties(device) }
            .iter()
            .any(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        {
            continue;
        }
        let memory = unsafe { instance.get_physical_device_memory_properties(device) };
        let local_bytes = memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .fold(0_u64, |total, h| total.saturating_add(h.size));
        let memory_mb = local_bytes / (1024 * 1024);
        // The driver's remaining process budget accounts for GPU memory
        // pressure, including allocations from other applications. If the
        // extension is absent, use a conservative cross-vendor fallback.
        let driver_free_mb = linux_vulkan_free_mb(
            &instance, device, &memory, supports_properties2,
        );
        let budget_mb = linux_offload_budget_mb(
            kind, memory_mb, available_ram_mb, driver_free_mb,
        );
        result.push(LinuxGpu {
            index, name, kind,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            memory_mb, budget_mb,
        });
    }
    unsafe { instance.destroy_instance(None) };
    result
}

// Linux Vulkan device indices refer to the native vkEnumeratePhysicalDevices
// order. GGML_VK_VISIBLE_DEVICES uses that same order, *not* ggml's backend
// device indices (which may include non-Vulkan backends).
#[cfg(target_os = "linux")]
fn choose_linux_gpu(devices: &[LinuxGpu], requested: Option<usize>) -> Option<&LinuxGpu> {
    devices.iter()
        .filter(|d| d.budget_mb > 0 && requested.map_or(true, |idx| d.index == idx))
        .max_by_key(|d| {
            let rank = match d.kind {
                "discrete" => 3_u8,
                "integrated" => 2,
                "virtual" => 1,
                _ => 0,
            };
            // Stable tie breaking avoids random GPU switching between runs.
            (rank, d.budget_mb, std::cmp::Reverse(d.index))
        })
}

#[cfg(target_os = "linux")]
fn first_visible_vulkan_index(raw: &str) -> Option<usize> {
    raw.split(|c: char| c == ',' || c.is_ascii_whitespace())
        .find(|p| !p.is_empty())?
        .parse::<usize>()
        .ok()
}

/// Called once, before backend initialization AND before worker threads start.
/// Allows selecting either an iGPU or dGPU while preserving an explicit GGML
/// override (including an empty value, which intentionally means CPU-only).
/// CHATY_VULKAN_DEVICE accepts the Vulkan physical-device index; by default
/// prefer the largest usable discrete GPU, then the largest integrated GPU.
#[cfg(target_os = "linux")]
pub fn configure_linux_vulkan_device() {
    if !cfg!(feature = "gpu") {
        return;
    }
    if std::env::var_os("GGML_VK_VISIBLE_DEVICES").is_some() {
        eprintln!("[chaty/linux] honoring existing GGML_VK_VISIBLE_DEVICES");
        return;
    }
    let devices = linux_vulkan_gpus();
    let forced = std::env::var("CHATY_VULKAN_DEVICE").ok();
    let selected = if let Some(ref value) = forced {
        value.trim().parse::<usize>().ok()
            .and_then(|idx| choose_linux_gpu(&devices, Some(idx)))
    } else {
        choose_linux_gpu(&devices, None)
    };
    match selected {
        Some(dev) => {
            // GGML reads this once, when the Vulkan backend is initialized.
            std::env::set_var("GGML_VK_VISIBLE_DEVICES", dev.index.to_string());
            eprintln!(
                "[chaty/linux] Vulkan device {}: {} ({}, budget {} MiB)",
                dev.index, dev.name, dev.kind, dev.budget_mb,
            );
        }
        None => {
            // Invalid explicit index must NOT silently select another GPU.
            // Also prevents a Vulkan software renderer from becoming an
            // accidental fallback when no compatible hardware was found.
            std::env::set_var("GGML_VK_VISIBLE_DEVICES", "");
            eprintln!(
                "[chaty/linux] no usable Vulkan GPU for selection {:?}; CPU fallback",
                forced,
            );
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_gpu_info(device: &LinuxGpu) -> GpuInfo {
    GpuInfo {
        name: device.name.clone(),
        vram_mb: device.budget_mb,
        dedicated_vram_mb: (device.kind == "discrete").then_some(device.memory_mb),
        kind: Some(device.kind.to_string()),
    }
}

#[cfg(target_os = "linux")]
pub fn detect_gpu() -> Option<GpuInfo> {
    if !cfg!(feature = "gpu") {
        return None;
    }
    let devices = linux_vulkan_gpus();
    // The visible device is authoritative. It is important that the auto-tuner
    // sizes layers for the SAME GPU on which GGML actually places them.
    let best = if let Ok(raw) = std::env::var("GGML_VK_VISIBLE_DEVICES") {
        choose_linux_gpu(&devices, Some(first_visible_vulkan_index(&raw)?))
    } else if let Ok(raw) = std::env::var("CHATY_VULKAN_DEVICE") {
        choose_linux_gpu(&devices, Some(raw.trim().parse::<usize>().ok()?))
    } else {
        choose_linux_gpu(&devices, None)
    }?;
    Some(linux_gpu_info(best))
}

#[cfg(all(test, target_os = "linux"))]
mod linux_gpu_selection_tests {
    use super::{choose_linux_gpu, first_visible_vulkan_index, linux_gpu_info, LinuxGpu};

    fn sample(index: usize, kind: &'static str, budget_mb: u64) -> LinuxGpu {
        LinuxGpu {
            index, name: format!("test-{index}"), kind, vendor_id: 0,
            device_id: 0, memory_mb: budget_mb, budget_mb,
        }
    }

    #[test]
    fn dedicated_capacity_is_distinct_from_runtime_budget() {
        let mut gpu = sample(0, "discrete", 6_000);
        gpu.memory_mb = 8_192;
        let info = linux_gpu_info(&gpu);
        assert_eq!(info.vram_mb, 6_000);
        assert_eq!(info.dedicated_vram_mb, Some(8_192));
        let json = serde_json::to_value(info).unwrap();
        assert_eq!(json["dedicatedVramMb"], 8_192);
        let integrated = linux_gpu_info(&sample(1, "integrated", 3_000));
        assert_eq!(integrated.dedicated_vram_mb, None);
    }

    #[test]
    fn discrete_is_preferred_even_if_integrated_has_larger_heap() {
        let devices = [sample(0, "integrated", 32_768), sample(1, "discrete", 8_192)];
        assert_eq!(choose_linux_gpu(&devices, None).unwrap().index, 1);
    }

    #[test]
    fn an_integrated_gpu_is_usable_when_no_discrete_gpu_exists() {
        let devices = [sample(2, "integrated", 2_048)];
        assert_eq!(choose_linux_gpu(&devices, None).unwrap().index, 2);
    }

    #[test]
    fn explicit_choice_can_select_integrated_gpu() {
        let devices = [sample(0, "integrated", 4_096), sample(1, "discrete", 8_192)];
        assert_eq!(choose_linux_gpu(&devices, Some(0)).unwrap().index, 0);
        assert!(choose_linux_gpu(&devices, Some(5)).is_none());
    }

    #[test]
    fn visibility_uses_first_physical_vulkan_index() {
        assert_eq!(first_visible_vulkan_index("2, 1"), Some(2));
        assert_eq!(first_visible_vulkan_index(""), None);
        assert_eq!(first_visible_vulkan_index("invalid"), None);
    }
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

/// macOS unified memory: "VRAM" is shared system memory, so report the WHOLE
/// device's memory usage (every app, plus the OS), not just this process. That
/// matches what users expect from a memory gauge and reflects real pressure.
#[cfg(target_os = "macos")]
pub fn gpu_usage() -> Option<GpuUsage> {
    // Reuse one System across the panel's 1.5s polls instead of allocating a
    // fresh one each call; just refresh the memory stats.
    use std::sync::{Mutex, OnceLock};
    static SYS: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let mut sys = SYS
        .get_or_init(|| Mutex::new(sysinfo::System::new()))
        .lock()
        .ok()?;
    sys.refresh_memory();
    let total_mb = sys.total_memory() / (1024 * 1024);
    let used_mb = sys.used_memory() / (1024 * 1024);
    if total_mb == 0 {
        return None;
    }
    Some(GpuUsage {
        used_mb: used_mb.min(total_mb),
        total_mb,
    })
}

#[cfg(not(any(windows, target_os = "macos")))]
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

    // Apple Silicon: unified memory, no PCIe copy, no separate VRAM. If the
    // weights comfortably fit in the working-set budget, offload EVERY layer —
    // that's the big win here. The OOM back-off in llama.rs::load() protects us
    // if this estimate is optimistic (it also has to hold the KV cache).
    #[cfg(target_os = "macos")]
    {
        let budget = vram_mb as f64 * 1024.0 * 1024.0;
        if (file_bytes as f64) < budget * 0.85 {
            return n_layer as i32 + 1;
        }
    }

    // What is actually FREE, not what the card has. Sizing from the total is how
    // a 26B model gets told it fits on a 12 GB card that a desktop, a browser
    // and a game are already sitting on — and a Vulkan driver answers that by
    // taking the process down rather than returning an allocation failure the
    // back-off could catch (issue #9). Falls back to the total when live usage
    // cannot be read, which is the old behaviour.
    let free_mb = gpu_usage()
        .filter(|u| u.total_mb > 0 && u.used_mb <= u.total_mb)
        .map_or(vram_mb, |u| u.total_mb.saturating_sub(u.used_mb));
    fit_layers(file_bytes, n_layer, free_mb.min(vram_mb) * 1024 * 1024)
}

/// How many of a model's layers fit in `free_bytes` of GPU memory.
///
/// The sizing rule on its own, so that everything that puts weights on a GPU
/// sizes them the same way — the chat model through `auto_gpu_layers`, the
/// knowledge-base embedder on every platform. Two copies of a rule is how one
/// of them ends up asking a full card for everything it has (issue #13).
///
/// Weights are taken as evenly spread across blocks (+1 for embeddings and the
/// output head), and ~20% — at least 1 GiB — is held back for the KV cache,
/// compute buffers and everything else already living there.
pub fn fit_layers(file_bytes: u64, n_layer: u32, free_bytes: u64) -> i32 {
    if n_layer == 0 || file_bytes == 0 {
        return 0;
    }
    let per_layer = file_bytes as f64 / (n_layer as f64 + 1.0);
    let vram = free_bytes as f64;
    let reserve = (vram * 0.20).max(1024.0 * 1024.0 * 1024.0);
    let budget = (vram - reserve).max(0.0);
    let fit = (budget / per_layer).floor() as i64;
    fit.clamp(0, n_layer as i64 + 1) as i32
}

/// What Metal can still hand this process on Apple Silicon: the device's
/// recommended working set minus what is already allocated on it — a chat
/// model's weights among it. The unified-memory counterpart of DXGI's budget
/// minus current usage on Windows.
///
/// Only this process's allocations are counted, and an MLX model lives in its
/// own sidecar process, so on its own this can read far roomier than the
/// machine is. It is therefore never used alone: see `embed_free_bytes`.
#[cfg(target_os = "macos")]
pub fn metal_free_bytes() -> Option<u64> {
    let d = metal::Device::system_default()?;
    let total = d.recommended_max_working_set_size();
    let used = d.current_allocated_size() as u64;
    Some(total.saturating_sub(used))
}

/// Free memory a model loaded NOW can use on the GPU.
///
/// On a discrete card that is the card's free VRAM. On Apple Silicon the GPU
/// and CPU share one pool, so two things must both have room: Metal's working
/// set (what this process may still allocate) and the machine's free memory
/// (which also sees a model held by the MLX sidecar). The smaller one binds.
pub fn embed_free_bytes(vram_mb: u64) -> u64 {
    #[cfg(target_os = "macos")]
    {
        let metal = metal_free_bytes().unwrap_or(vram_mb * 1024 * 1024);
        let ram = gpu_usage()
            .filter(|u| u.total_mb > 0 && u.used_mb <= u.total_mb)
            .map_or(u64::MAX, |u| (u.total_mb - u.used_mb) * 1024 * 1024);
        metal.min(ram)
    }
    #[cfg(not(target_os = "macos"))]
    {
        gpu_usage()
            .filter(|u| u.total_mb > 0 && u.used_mb <= u.total_mb)
            .map_or(vram_mb, |u| u.total_mb.saturating_sub(u.used_mb))
            .min(vram_mb)
            * 1024
            * 1024
    }
}

#[cfg(test)]
mod tests {
    /// One sizing rule for every GPU load. Pinned on the embedder's own numbers
    /// (bge-m3 Q8_0: 634 MB, 24 blocks) because that is the load issue #13 was
    /// about, and on a chat model to show the extraction changed nothing there.
    #[test]
    fn fit_layers_sizes_against_what_is_free() {
        use super::fit_layers;
        const GIB: u64 = 1024 * 1024 * 1024;
        let bge = 634_553_760u64;

        assert_eq!(fit_layers(bge, 24, 2 * GIB), 25, "room to spare: every layer");
        assert_eq!(fit_layers(bge, 24, GIB + GIB / 5), 8, "a sliver over the reserve: part of it");
        assert_eq!(fit_layers(bge, 24, GIB), 0, "nothing past the 1 GiB held back: the CPU");
        assert_eq!(fit_layers(bge, 24, 0), 0);
        assert_eq!(fit_layers(0, 24, 8 * GIB), 0, "no file");
        assert_eq!(fit_layers(bge, 0, 8 * GIB), 0, "no layers");

        // Never fewer layers for more room.
        let mut last = 0;
        for mb in (0..8192).step_by(64) {
            let n = fit_layers(bge, 24, mb * 1024 * 1024);
            assert!(n >= last, "more free memory gave fewer layers at {mb} MB");
            last = n;
        }

        // A 9B at Q4 (~5.5 GB, 32 blocks) on an 8 GB card with nothing else on
        // it gets all of itself, as it did before the rule had a name.
        assert_eq!(fit_layers(5_500_000_000, 32, 8 * GIB), 33);
    }
}
