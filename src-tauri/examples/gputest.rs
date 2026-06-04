//! Headless GPU offload diagnostic.
//!
//!   cargo run --release --example gputest -- <model.gguf>
//!
//! Prints the detected GPU, the auto-tuned layer count, and then loads the
//! model WITHOUT muting llama.cpp's logs so you can see whether the Vulkan
//! backend found a device and how many layers it actually offloaded.

use anyhow::{Context, Result};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: gputest <model.gguf>")?;

    println!("--- detect_gpu (DXGI) ---");
    let gpu = chaty_lib::gpu::detect_gpu();
    match &gpu {
        Some(g) => println!("GPU: {}  |  {} MB VRAM", g.name, g.vram_mb),
        None => println!("No discrete GPU detected"),
    }

    // Keep llama.cpp logs ON so we can see Vulkan device discovery + offload.
    let backend = LlamaBackend::init()?;

    println!("\n--- probe n_layer (vocab_only + block_count metadata) ---");
    let probe = LlamaModel::load_from_file(
        &backend,
        &path,
        &LlamaModelParams::default().with_vocab_only(true),
    );
    let n_layer = match &probe {
        Ok(m) => {
            let via_n_layer = m.n_layer();
            let via_meta = m
                .meta_val_str("general.architecture")
                .ok()
                .and_then(|arch| m.meta_val_str(&format!("{arch}.block_count")).ok())
                .and_then(|s| s.trim().parse::<u32>().ok());
            println!("n_layer() = {via_n_layer}  |  block_count meta = {via_meta:?}");
            via_meta.unwrap_or(via_n_layer)
        }
        Err(e) => {
            println!("vocab_only probe FAILED: {e:#}");
            0
        }
    };

    let vram = gpu.as_ref().map(|g| g.vram_mb).unwrap_or(0);
    let auto = chaty_lib::gpu::auto_gpu_layers(std::path::Path::new(&path), n_layer, vram);
    println!("\nauto_gpu_layers = {auto}  (of {n_layer} layers)");

    println!("\n--- loading with n_gpu_layers={auto} (watch for 'Vulkan' / 'offloaded ...') ---");
    match LlamaModel::load_from_file(
        &backend,
        &path,
        &LlamaModelParams::default().with_n_gpu_layers(auto.max(0) as u32),
    ) {
        Ok(m) => println!("\nLOADED OK. model n_layer={}", m.n_layer()),
        Err(e) => println!("\nGPU LOAD FAILED: {e:#}"),
    }
    Ok(())
}
