//! llama.cpp's own account of what went wrong.
//!
//! Its log used to be voided wholesale, so a model it refused came back as a
//! null pointer and the reason had to be guessed from the file's header — and
//! was guessed wrong: a file whose weights use a quantization type this build
//! does not have was reported as "does not know the architecture qwen35", an
//! architecture this build knows perfectly well (issue #20). Warnings and
//! errors are kept now, in a small ring; info and debug lines are still
//! dropped, and everything is echoed to stderr under `CHATY_LLAMA_LOG=1`.

use std::collections::VecDeque;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::agent::tr;

const KEEP: usize = 64;

struct Ring {
    lines: VecDeque<(u64, String)>,
    next: u64,
    /// The message being written is one we keep — a `CONT` piece belongs to
    /// whatever came before it, dropped or kept.
    keeping: bool,
}

static RING: Mutex<Ring> = Mutex::new(Ring { lines: VecDeque::new(), next: 0, keeping: false });
static ECHO: AtomicBool = AtomicBool::new(false);

fn record(level: llama_cpp_sys_2::ggml_log_level, piece: &str) {
    let Ok(mut r) = RING.lock() else { return };
    if level == llama_cpp_sys_2::GGML_LOG_LEVEL_CONT {
        if r.keeping {
            if let Some((_, last)) = r.lines.back_mut() {
                last.push_str(piece.trim_end_matches('\n'));
            }
        }
        return;
    }
    r.keeping =
        level == llama_cpp_sys_2::GGML_LOG_LEVEL_WARN || level == llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR;
    if !r.keeping {
        return;
    }
    let seq = r.next;
    r.next += 1;
    r.lines.push_back((seq, piece.trim_end_matches('\n').to_string()));
    while r.lines.len() > KEEP {
        r.lines.pop_front();
    }
}

unsafe extern "C" fn capture(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const c_char,
    _user: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp hands a NUL-terminated string that lives for the call.
    let piece = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    if ECHO.load(Ordering::Relaxed) {
        eprint!("{piece}");
    }
    record(level, &piece);
}

/// Route llama.cpp's (and ggml's) log through the ring. Replaces
/// `LlamaBackend::void_logs`.
pub fn install() {
    ECHO.store(std::env::var("CHATY_LLAMA_LOG").as_deref() == Ok("1"), Ordering::Relaxed);
    // SAFETY: a plain C callback with no user data; llama_log_set also points
    // ggml's logger at it.
    unsafe { llama_cpp_sys_2::llama_log_set(Some(capture), std::ptr::null_mut()) };
}

/// Where the log stands now — pass it to [`since`] after the call to learn
/// what that call said.
pub fn mark() -> u64 {
    RING.lock().map(|r| r.next).unwrap_or(0)
}

/// The warnings and errors logged since `mark`, oldest first.
pub fn since(mark: u64) -> Vec<String> {
    RING.lock()
        .map(|r| r.lines.iter().filter(|(s, _)| *s >= mark).map(|(_, l)| l.clone()).collect())
        .unwrap_or_default()
}

/// Turn what llama.cpp said while refusing a model into something a person can
/// act on. The two refusals that have a clear remedy are named; anything else
/// is passed on in llama.cpp's own words rather than guessed at.
pub fn explain_refusal(said: &[String]) -> Option<String> {
    for line in said {
        if let Some(rest) = line.split("unknown model architecture: '").nth(1) {
            let arch = rest.split('\'').next().unwrap_or(rest);
            return Some(trf!(
                "这个 llama.cpp 版本不认识模型架构 \"{arch}\"。需要更新 Chaty 才能加载它;如果这个模型有 MLX 版本,在 Mac 上可以先用那个。",
                "this llama.cpp build does not know the model architecture \"{arch}\". Chaty needs an update to load it; on a Mac, an MLX build of the same model may work in the meantime."
            ));
        }
        if let Some(rest) = line.split("has invalid ggml type ").nth(1) {
            let n: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            let bound: String = rest
                .split("should be in [0, ")
                .nth(1)
                .map(|b| b.chars().take_while(|c| c.is_ascii_digit()).collect())
                .unwrap_or_default();
            let known = if bound.is_empty() { String::new() } else { format!(" (0–{})", bound.parse::<u32>().map(|b| b.saturating_sub(1)).unwrap_or(0)) };
            return Some(trf!(
                "模型的权重用了这个 llama.cpp 不支持的量化类型(编号 {n},本版本支持{known})。这通常是某个 llama.cpp 分支专用的格式——比如一些三值/1-bit 量化的定制版本——只能用那个分支运行。换一个标准量化(如 Q4_K_M、Q8_0)的 GGUF 就可以。",
                "the model's weights use a quantization type this llama.cpp does not support (type {n}; this build supports{known}). That is usually a format specific to one llama.cpp fork — some ternary or 1-bit builds are — and only that fork can run it. A GGUF in a standard quantization (Q4_K_M, Q8_0, …) will load."
            ));
        }
    }
    // Nothing we recognise: llama.cpp's own last word, which is still better
    // than a null pointer or a guess.
    let last: Vec<&String> = said.iter().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
    if last.is_empty() {
        return None;
    }
    let words = last.iter().map(|l| l.trim()).collect::<Vec<_>>().join(" / ");
    Some(format!("{}{words}", tr("llama.cpp 报告:", "llama.cpp says: ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_is_explained_from_what_llama_cpp_said() {
        let arch = explain_refusal(&["llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'glimmer9'".into()]).unwrap();
        assert!(arch.contains("glimmer9"), "{arch}");

        // Issue #20: an unsupported quant type, NOT an unknown architecture.
        let quant = explain_refusal(&["gguf_init_from_file_impl: tensor 'blk.0.attn_q.weight' has invalid ggml type 57. should be in [0, 43)".into()]).unwrap();
        assert!(quant.contains("57"), "{quant}");
        assert!(quant.contains("42"), "names the range this build has: {quant}");
        assert!(!quant.contains("架构") && !quant.contains("architecture"), "{quant}");

        // Anything else is passed on as said, not guessed at.
        let other = explain_refusal(&["llama_model_load: error: something new went wrong".into()]).unwrap();
        assert!(other.contains("something new went wrong"), "{other}");
        assert!(explain_refusal(&[]).is_none());
    }

    #[test]
    fn continuation_pieces_join_the_line_they_continue() {
        let start = mark();
        record(llama_cpp_sys_2::GGML_LOG_LEVEL_INFO, "loading tensors\n");
        record(llama_cpp_sys_2::GGML_LOG_LEVEL_CONT, " ...ignored with its info line\n");
        record(llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR, "load failed:");
        record(llama_cpp_sys_2::GGML_LOG_LEVEL_CONT, " bad tensor\n");
        assert_eq!(since(start), vec!["load failed: bad tensor".to_string()]);
    }
}
