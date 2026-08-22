//! Native microphone capture (used on macOS).
//!
//! WKWebView never exposes capture devices to embedded apps — `getUserMedia`
//! fails with "no device was found amongst 0 devices" regardless of TCC,
//! entitlements, or WebKit preference flags — so on macOS the frontend records
//! through these commands instead. A dedicated thread owns the cpal stream
//! (cpal streams are !Send); commands talk to it through channels and shared
//! buffers. Mono f32 samples are returned base64-encoded (little-endian), the
//! same wire format the web recorder feeds to `transcribe`.
//!
//! On Windows/Linux the web recorder works fine, so these are stubs there.

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MicResult {
    /// Base64 of little-endian f32 mono PCM (same encoding as the web path).
    pub samples: String,
    pub sample_rate: u32,
}

/// Start capturing from the default input device. Returns the sample rate.
#[tauri::command]
pub fn mic_start() -> Result<u32, String> {
    #[cfg(target_os = "macos")]
    {
        imp::start()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("native capture is only used on macOS".into())
    }
}

/// RMS level of the latest input block (0.0–1.0-ish), for the orb/VAD.
#[tauri::command]
pub fn mic_level() -> f32 {
    #[cfg(target_os = "macos")]
    {
        imp::level()
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}

/// Stop capturing and return everything recorded since `mic_start`.
#[tauri::command]
pub fn mic_stop() -> Result<MicResult, String> {
    #[cfg(target_os = "macos")]
    {
        imp::stop()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("native capture is only used on macOS".into())
    }
}

/// Stop capturing and discard the audio.
#[tauri::command]
pub fn mic_cancel() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        imp::cancel()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::MicResult;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Duration;

    use base64::Engine;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    #[derive(Default)]
    struct Shared {
        samples: Mutex<Vec<f32>>,
        /// RMS of the most recent input block (for the level meter / VAD).
        level: Mutex<f32>,
        sample_rate: AtomicU32,
    }

    static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();
    struct CaptureControl {
        stop: Sender<()>,
        /// Signalled after the capture stream has actually released CoreAudio.
        released: Receiver<()>,
    }

    static STOP: Mutex<Option<CaptureControl>> = Mutex::new(None);

    fn shared() -> &'static Arc<Shared> {
        SHARED.get_or_init(|| Arc::new(Shared::default()))
    }

    pub fn start() -> Result<u32, String> {
        // App-level TCC consent first — this is the prompt the user can grant.
        if !crate::voice::request_mic_permission() {
            return Err(
                "麦克风权限被拒绝，请在 系统设置 → 隐私与安全性 → 麦克风 中允许 Chaty。(Microphone access denied — allow Chaty in System Settings → Privacy & Security → Microphone.)".into(),
            );
        }

        let mut stop_slot = STOP.lock().unwrap();
        if stop_slot.is_some() {
            return Err("已经在录音了 (already recording)".into());
        }

        let sh = shared().clone();
        sh.samples.lock().unwrap().clear();
        *sh.level.lock().unwrap() = 0.0;

        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let (released_tx, released_rx) = std::sync::mpsc::channel::<()>();
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<u32, String>>();

        std::thread::Builder::new()
            .name("chaty-mic".into())
            .spawn(move || {
                let host = cpal::default_host();
                // The default device can be a phantom (Continuity iPhone mic,
                // monitor mic that's powered off, aggregate devices) whose
                // config query fails — e.g. on a Mac mini with no built-in
                // microphone. Scan until one actually works.
                let (device, config) = match pick_input(&host) {
                    Ok(dc) => dc,
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };
                eprintln!(
                    "mic: using '{}' ({:?} @ {} Hz, {} ch)",
                    device.name().unwrap_or_default(),
                    config.sample_format(),
                    config.sample_rate().0,
                    config.channels()
                );
                let sample_rate = config.sample_rate().0;
                let channels = (config.channels() as usize).max(1);
                let sh_cb = sh.clone();

                let on_err = |e| eprintln!("mic stream error: {e}");
                let push = move |mono: &mut dyn Iterator<Item = f32>, n: usize| {
                    let mut sum = 0.0f32;
                    {
                        let mut buf = sh_cb.samples.lock().unwrap();
                        for s in mono {
                            buf.push(s);
                            sum += s * s;
                        }
                    }
                    *sh_cb.level.lock().unwrap() = (sum / n.max(1) as f32).sqrt();
                };
                use cpal::SampleFormat;
                let fmt = config.sample_format();
                let cfg: cpal::StreamConfig = config.into();
                let stream = match fmt {
                    SampleFormat::F32 => device.build_input_stream(
                        &cfg,
                        {
                            let push = push.clone();
                            move |data: &[f32], _| {
                                let n = data.len() / channels;
                                push(&mut data.chunks(channels).map(|f| f[0]), n)
                            }
                        },
                        on_err,
                        None,
                    ),
                    SampleFormat::I16 => device.build_input_stream(
                        &cfg,
                        {
                            let push = push.clone();
                            move |data: &[i16], _| {
                                let n = data.len() / channels;
                                push(&mut data.chunks(channels).map(|f| f[0] as f32 / 32768.0), n)
                            }
                        },
                        on_err,
                        None,
                    ),
                    SampleFormat::U16 => device.build_input_stream(
                        &cfg,
                        {
                            let push = push.clone();
                            move |data: &[u16], _| {
                                let n = data.len() / channels;
                                push(
                                    &mut data
                                        .chunks(channels)
                                        .map(|f| (f[0] as f32 - 32768.0) / 32768.0),
                                    n,
                                )
                            }
                        },
                        on_err,
                        None,
                    ),
                    other => {
                        let _ = init_tx.send(Err(format!(
                            "不支持的采样格式 (unsupported sample format): {other:?}"
                        )));
                        return;
                    }
                };
                let stream = match stream {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = init_tx.send(Err(format!(
                            "无法打开麦克风输入流 (failed to open input stream): {e}"
                        )));
                        return;
                    }
                };
                if let Err(e) = stream.play() {
                    let _ = init_tx.send(Err(format!(
                        "无法启动麦克风输入流 (failed to start input stream): {e}"
                    )));
                    return;
                }
                sh.sample_rate.store(sample_rate, Ordering::Relaxed);
                let _ = init_tx.send(Ok(sample_rate));
                // Hold the stream until asked to stop (or the sender is dropped).
                let _ = stop_rx.recv();
                drop(stream);
                let _ = released_tx.send(());
            })
            .map_err(|e| e.to_string())?;

        match init_rx.recv() {
            Ok(Ok(rate)) => {
                *stop_slot = Some(CaptureControl {
                    stop: stop_tx,
                    released: released_rx,
                });
                Ok(rate)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err("录音线程启动失败 (capture thread failed to start)".into()),
        }
    }

    pub fn level() -> f32 {
        *shared().level.lock().unwrap()
    }

    pub fn stop() -> Result<MicResult, String> {
        signal_stop()?;
        let sh = shared();
        let samples = std::mem::take(&mut *sh.samples.lock().unwrap());
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for s in &samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Ok(MicResult {
            samples: base64::engine::general_purpose::STANDARD.encode(bytes),
            sample_rate: sh.sample_rate.load(Ordering::Relaxed),
        })
    }

    pub fn cancel() -> Result<(), String> {
        signal_stop()?;
        shared().samples.lock().unwrap().clear();
        Ok(())
    }

    /// First input device whose configuration is actually readable.
    fn pick_input(
        host: &cpal::Host,
    ) -> Result<(cpal::Device, cpal::SupportedStreamConfig), String> {
        let mut tried: Vec<String> = Vec::new();
        let mut try_dev = |d: cpal::Device| -> Option<(cpal::Device, cpal::SupportedStreamConfig)> {
            let name = d.name().unwrap_or_else(|_| "?".into());
            match d.default_input_config() {
                Ok(c) => return Some((d, c)),
                Err(e) => tried.push(format!("{name}: {e}")),
            }
            if let Ok(mut cfgs) = d.supported_input_configs() {
                if let Some(c) = cfgs.next() {
                    return Some((d, c.with_max_sample_rate()));
                }
            }
            None
        };
        if let Some(d) = host.default_input_device() {
            if let Some(dc) = try_dev(d) {
                return Ok(dc);
            }
        }
        if let Ok(devices) = host.input_devices() {
            for d in devices {
                if let Some(dc) = try_dev(d) {
                    return Ok(dc);
                }
            }
        }
        Err(format!(
            "没有可用的麦克风输入设备 (no usable microphone)。已尝试 (tried): {}",
            if tried.is_empty() {
                "系统未报告任何输入设备 (system reports no input devices)".to_string()
            } else {
                tried.join("; ")
            }
        ))
    }

    fn signal_stop() -> Result<(), String> {
        // Do not hold STOP while waiting. The acknowledgement is the real
        // device-release handshake; the timeout only prevents a broken audio
        // backend from hanging the command forever.
        let control = STOP
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| "当前没有进行中的录音 (no recording in progress)".to_string())?;
        let _ = control.stop.send(());
        match control.released.recv_timeout(Duration::from_secs(2)) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => Ok(()),
            Err(RecvTimeoutError::Timeout) => {
                eprintln!("mic: timed out waiting for capture thread to release the device");
                Ok(())
            }
        }
    }
}
