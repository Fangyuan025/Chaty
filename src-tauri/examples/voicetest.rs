//! Headless round-trip check: Kokoro TTS -> Whisper STT.
//! Downloads ~0.5 GB of models on first run (cached in temp).
//!   cargo run --example voicetest

use anyhow::Result;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("chaty-voice-models");
    eprintln!("models dir: {}", dir.display());

    let text = "Hello, this is Chaty speaking.";
    eprintln!("synthesizing: {text:?}");
    let (samples, sr) = chaty_lib::voice::synthesize(dir.clone(), text.into(), 1.0, 0).await?;
    eprintln!(
        "TTS -> {} samples @ {} Hz ({:.2}s)",
        samples.len(),
        sr,
        samples.len() as f32 / sr as f32
    );

    let wav = dir.join("tts_out.wav");
    let _ = sherpa_rs::write_audio_file(wav.to_str().unwrap(), &samples, sr);
    eprintln!("wrote {} (listen to verify TTS)", wav.display());

    eprintln!("transcribing the synthesized audio back...");
    let recognized = chaty_lib::voice::transcribe(dir, samples, sr).await?;
    println!("\nSTT result: {recognized:?}");
    Ok(())
}
