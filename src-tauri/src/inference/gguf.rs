//! Reading a GGUF's header — its metadata keys and its tensor table — without
//! loading the model.
//!
//! Two callers need to look inside a file before (or instead of) handing it to
//! llama.cpp: the load-failure diagnosis, which explains a file the loader
//! refused, and the image-model probe, which recognises a diffusion model that
//! llama.cpp was never going to load. Diffusion GGUFs are the reason the tensor
//! table is read at all: the ones in circulation often carry no metadata keys
//! whatsoever, and their tensor names are the only thing that says what they
//! are.

use std::io::Read;

fn take<R: Read>(r: &mut R, n: usize) -> Option<Vec<u8>> {
    let mut b = vec![0u8; n];
    r.read_exact(&mut b).ok()?;
    Some(b)
}
fn u32le<R: Read>(r: &mut R) -> Option<u32> {
    Some(u32::from_le_bytes(take(r, 4)?.try_into().ok()?))
}
fn u64le<R: Read>(r: &mut R) -> Option<u64> {
    Some(u64::from_le_bytes(take(r, 8)?.try_into().ok()?))
}
fn string<R: Read>(r: &mut R) -> Option<String> {
    let n = u64le(r)?;
    // A length this large is a malformed file, not a long key.
    if n > 1 << 20 {
        return None;
    }
    String::from_utf8(take(r, n as usize)?).ok()
}
/// Read past a value without keeping it.
fn skip_value<R: Read>(r: &mut R, t: u32) -> Option<()> {
    match t {
        0 | 1 | 7 => take(r, 1).map(|_| ()),
        2 | 3 => take(r, 2).map(|_| ()),
        4..=6 => take(r, 4).map(|_| ()),
        10..=12 => take(r, 8).map(|_| ()),
        8 => string(r).map(|_| ()),
        9 => {
            let et = u32le(r)?;
            let n = u64le(r)?;
            if n > 8_000_000 {
                return None;
            }
            for _ in 0..n {
                skip_value(r, et)?;
            }
            Some(())
        }
        _ => None,
    }
}

/// One entry of the tensor table.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
}

/// What a GGUF says about itself before its weights begin.
#[derive(Debug, Default)]
pub struct Header {
    /// `general.architecture`, when present and a string.
    pub arch: Option<String>,
    /// Every metadata key read, in file order.
    pub keys: Vec<String>,
    /// The tensor table — empty unless it was asked for, and never read when
    /// the metadata already showed a tokenizer (see [`read_header`]).
    pub tensors: Vec<Tensor>,
    /// The file carries `tokenizer.*` metadata: a language model's GGUF.
    pub has_tokenizer: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum HeaderError {
    /// The file does not start with the GGUF magic.
    NotGguf,
    /// It does, but what follows could not be read.
    Malformed,
}

/// Read a GGUF header.
///
/// `tensors` also reads the tensor table. `stop_at_tokenizer` ends the read at
/// the first `tokenizer.*` key: a file that has one is a language model, and
/// the vocabulary arrays that follow are the largest part of any header — the
/// question "is this an image model" is answered without reading them.
pub fn read_header<R: Read>(
    mut r: R,
    tensors: bool,
    stop_at_tokenizer: bool,
) -> Result<Header, HeaderError> {
    use HeaderError::Malformed;
    let magic = take(&mut r, 4).ok_or(Malformed)?;
    if magic != b"GGUF" {
        return Err(HeaderError::NotGguf);
    }
    let _version = u32le(&mut r).ok_or(Malformed)?;
    let n_tensors = u64le(&mut r).ok_or(Malformed)?;
    let n_kv = u64le(&mut r).ok_or(Malformed)?;
    if n_kv > 100_000 || n_tensors > 1_000_000 {
        return Err(Malformed);
    }

    let mut h = Header::default();
    for _ in 0..n_kv {
        let key = string(&mut r).ok_or(Malformed)?;
        let t = u32le(&mut r).ok_or(Malformed)?;
        if key.starts_with("tokenizer.") {
            h.has_tokenizer = true;
            if stop_at_tokenizer {
                h.keys.push(key);
                return Ok(h);
            }
        }
        if key == "general.architecture" && t == 8 {
            h.arch = string(&mut r);
        } else {
            skip_value(&mut r, t).ok_or(Malformed)?;
        }
        h.keys.push(key);
    }

    if tensors {
        h.tensors.reserve(n_tensors.min(100_000) as usize);
        for _ in 0..n_tensors {
            let name = string(&mut r).ok_or(Malformed)?;
            let n_dims = u32le(&mut r).ok_or(Malformed)?;
            if n_dims > 8 {
                return Err(Malformed);
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(u64le(&mut r).ok_or(Malformed)?);
            }
            let _ggml_type = u32le(&mut r).ok_or(Malformed)?;
            let _offset = u64le(&mut r).ok_or(Malformed)?;
            h.tensors.push(Tensor { name, dims });
        }
    }
    Ok(h)
}

/// [`read_header`] on a file, buffered.
pub fn read_header_file(
    path: &std::path::Path,
    tensors: bool,
    stop_at_tokenizer: bool,
) -> Result<Header, HeaderError> {
    let f = std::fs::File::open(path).map_err(|_| HeaderError::Malformed)?;
    read_header(std::io::BufReader::with_capacity(1 << 16, f), tensors, stop_at_tokenizer)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    /// A minimal GGUF: string metadata, then a tensor table of 1-D F32 tensors.
    pub(crate) fn gguf_bytes(kv: &[(&str, &str)], tensors: &[(&str, &[u64])]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        let s = |b: &mut Vec<u8>, v: &str| {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        };
        for (k, v) in kv {
            s(&mut b, k);
            b.extend_from_slice(&8u32.to_le_bytes());
            s(&mut b, v);
        }
        for (name, dims) in tensors {
            s(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        b
    }

    #[test]
    fn reads_metadata_and_the_tensor_table() {
        let bytes = gguf_bytes(
            &[("general.architecture", "flux")],
            &[("double_blocks.0.img_attn.qkv.weight", &[3072, 9216]), ("img_in.weight", &[64, 3072])],
        );
        let h = read_header(Cursor::new(bytes), true, true).unwrap();
        assert_eq!(h.arch.as_deref(), Some("flux"));
        assert!(!h.has_tokenizer);
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.tensors[1].name, "img_in.weight");
        assert_eq!(h.tensors[1].dims, vec![64, 3072]);
    }

    /// A language model is recognised by its tokenizer, and the read stops
    /// there — before the vocabulary and before the tensor table.
    #[test]
    fn stops_at_the_first_tokenizer_key() {
        let bytes = gguf_bytes(
            &[("general.architecture", "qwen3"), ("tokenizer.ggml.model", "gpt2"), ("zzz", "never read")],
            &[("token_embd.weight", &[4, 4])],
        );
        let h = read_header(Cursor::new(bytes), true, true).unwrap();
        assert!(h.has_tokenizer);
        assert!(h.tensors.is_empty());
        assert_eq!(h.keys, vec!["general.architecture", "tokenizer.ggml.model"]);
    }

    #[test]
    fn a_file_that_is_not_gguf_says_so() {
        assert_eq!(read_header(Cursor::new(b"NOPE....".to_vec()), true, false).unwrap_err(), HeaderError::NotGguf);
        assert_eq!(read_header(Cursor::new(b"GGUF".to_vec()), true, false).unwrap_err(), HeaderError::Malformed);
    }
}
