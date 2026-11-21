//! A GGUF written in a layout llama.cpp does not read, rewritten once into the
//! one it does.
//!
//! Inkling 2 Mini runs a block of its middle layers three times. Unsloth's own
//! GGUFs of it — the files at the top of their repository, and others made
//! with their converter — describe that loop in a layout of their own:
//! architecture `inkling2`, the loop under `inkling2.num_loops` and
//! `inkling2.loop.core_start`/`core_end`, `block_count` counting every pass,
//! the loop's control tensors named `loop_attn_*`/`loop_mlp_*`, and the
//! message queries of all the loop's layers in two model-wide tensors.
//! Mainline llama.cpp reads it as `inkling2_loop` (#32694): `loop.start`,
//! `loop.end` and `loop.count`, `block_count` the physical layers,
//! `loop_q_ctrl_*`/`loop_ffn_ctrl_*`, and one `loop_msg_query` per looped
//! layer, its q, k, v and ffn queries side by side. The tensors themselves are
//! the same bytes, so the rewrite is a copy: what it writes is byte for byte
//! the file Unsloth publish for mainline (`Mainline_Support/…gguf_file`, the
//! same SHA-256).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// GGUF value types (gguf.h).
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;

fn fixed_size(t: u32) -> Option<u64> {
    Some(match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

/// One metadata entry, its value kept as the bytes after its type tag.
struct Kv {
    key: String,
    ty: u32,
    raw: Vec<u8>,
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    ty: u32,
    offset: u64,
}

struct Gguf {
    version: u32,
    kvs: Vec<Kv>,
    tensors: Vec<TensorInfo>,
    /// Where the tensor data starts in the file.
    data_start: u64,
    alignment: u64,
    file_len: u64,
}

fn u32le(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn u64le(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn bytes(r: &mut impl Read, n: u64, out: &mut Vec<u8>) -> Result<()> {
    if n > 1 << 32 {
        bail!("GGUF field of {n} bytes");
    }
    let at = out.len();
    out.resize(at + n as usize, 0);
    r.read_exact(&mut out[at..])?;
    Ok(())
}

fn string(r: &mut impl Read) -> Result<String> {
    let n = u64le(r)?;
    let mut b = Vec::new();
    bytes(r, n, &mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// The bytes of a value of type `ty`, as they follow its tag.
fn raw_value(r: &mut impl Read, ty: u32, out: &mut Vec<u8>) -> Result<()> {
    if let Some(n) = fixed_size(ty) {
        return bytes(r, n, out);
    }
    match ty {
        T_STRING => {
            let n = u64le(r)?;
            out.extend_from_slice(&n.to_le_bytes());
            bytes(r, n, out)
        }
        T_ARRAY => {
            let at = u32le(r)?;
            let n = u64le(r)?;
            out.extend_from_slice(&at.to_le_bytes());
            out.extend_from_slice(&n.to_le_bytes());
            if let Some(size) = fixed_size(at) {
                return bytes(r, n.checked_mul(size).context("GGUF array size")?, out);
            }
            for _ in 0..n {
                raw_value(r, at, out)?;
            }
            Ok(())
        }
        _ => bail!("GGUF value type {ty}"),
    }
}

fn read(path: &Path) -> Result<Gguf> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = u32le(&mut r)?;
    let n_tensors = u64le(&mut r)?;
    let n_kv = u64le(&mut r)?;
    let mut kvs = Vec::new();
    for _ in 0..n_kv {
        let key = string(&mut r)?;
        let ty = u32le(&mut r)?;
        let mut raw = Vec::new();
        raw_value(&mut r, ty, &mut raw)?;
        kvs.push(Kv { key, ty, raw });
    }
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let name = string(&mut r)?;
        let nd = u32le(&mut r)?;
        let dims = (0..nd).map(|_| u64le(&mut r)).collect::<Result<Vec<_>>>()?;
        let ty = u32le(&mut r)?;
        let offset = u64le(&mut r)?;
        tensors.push(TensorInfo { name, dims, ty, offset });
    }
    let header_end = r.stream_position()?;
    let mut g = Gguf { version, kvs, tensors, data_start: 0, alignment: 32, file_len };
    if let Some(a) = g.int("general.alignment") {
        g.alignment = a.max(1);
    }
    g.data_start = header_end.div_ceil(g.alignment) * g.alignment;
    Ok(g)
}

impl Gguf {
    fn kv(&self, key: &str) -> Option<&Kv> {
        self.kvs.iter().find(|k| k.key == key)
    }

    fn int(&self, key: &str) -> Option<u64> {
        let kv = self.kv(key)?;
        let b = &kv.raw;
        Some(match kv.ty {
            T_U32 | T_I32 => u32::from_le_bytes(b.get(..4)?.try_into().ok()?) as u64,
            T_U64 | T_I64 => u64::from_le_bytes(b.get(..8)?.try_into().ok()?),
            _ => return None,
        })
    }

    fn str(&self, key: &str) -> Option<String> {
        let kv = self.kv(key).filter(|k| k.ty == T_STRING)?;
        Some(String::from_utf8_lossy(kv.raw.get(8..)?).into_owned())
    }

    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The bytes from each tensor's start to the next one's (padding and all),
    /// by name — the data is copied in these spans.
    fn spans(&self) -> Vec<(usize, u64)> {
        let mut order: Vec<usize> = (0..self.tensors.len()).collect();
        order.sort_by_key(|&i| self.tensors[i].offset);
        let data_len = self.file_len.saturating_sub(self.data_start);
        order
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let end = order.get(k + 1).map_or(data_len, |&j| self.tensors[j].offset);
                (i, end.saturating_sub(self.tensors[i].offset))
            })
            .collect()
    }
}

/// What a rewrite into mainline's `inkling2_loop` takes, read from a file in
/// Unsloth's looped layout.
struct Loop {
    passes: u64,
    start: u64,
    end: u64,
    /// The layers that exist once: every pass counted, less the repeats.
    physical: u64,
    /// Bytes of one message query (one row of the model-wide tensors).
    row: u64,
}

const ARCH: &str = "inkling2";
const LOOP_ARCH: &str = "inkling2_loop";

fn plan(g: &Gguf) -> Option<Loop> {
    if g.str("general.architecture")?.as_str() != ARCH {
        return None;
    }
    let passes = g.int("inkling2.num_loops")?;
    let start = g.int("inkling2.loop.core_start")?;
    let end = g.int("inkling2.loop.core_end")?;
    let blocks = g.int("inkling2.block_count")?;
    let core = end.checked_sub(start).filter(|&c| c > 0)?;
    let physical = blocks.checked_sub(passes.checked_sub(1)?.checked_mul(core)?)?;
    let res = g.tensor("loop_res_query.weight")?;
    let kv = g.tensor("loop_kv_res_query.weight")?;
    if passes < 2 || end > physical || res.dims.len() != 2 || res.dims != kv.dims || res.ty != kv.ty || res.dims[1] != 2 * core {
        return None;
    }
    // F32, F16 or BF16: rows of plain elements, cut apart as they are.
    let elem = match res.ty {
        0 => 4,
        1 | 30 => 2,
        _ => return None,
    };
    Some(Loop { passes, start, end, physical, row: res.dims[0] * elem })
}

/// Whether the file at `path` is in a layout llama.cpp does not read but this
/// module can rewrite.
pub fn needs_rewrite(path: &Path) -> bool {
    read(path).ok().as_ref().and_then(plan).is_some()
}

/// The name a loop control tensor goes by in mainline.
fn mainline_name(name: &str) -> String {
    let Some(rest) = name.strip_prefix("blk.") else { return name.to_string() };
    let Some((layer, tail)) = rest.split_once('.') else { return name.to_string() };
    for (from, to) in [("loop_attn_", "loop_q_ctrl_"), ("loop_k_", "loop_k_ctrl_"), ("loop_mlp_", "loop_ffn_ctrl_"), ("loop_v_", "loop_v_ctrl_")] {
        if let Some(part) = tail.strip_prefix(from) {
            if matches!(part, "down.weight" | "forget.weight" | "gain.weight") {
                return format!("blk.{layer}.{to}{part}");
            }
        }
    }
    name.to_string()
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// The metadata as mainline has it: the same entries in the same order, under
/// the loop architecture's prefix, the loop renamed, the per-layer arrays and
/// the block count down to the physical layers, and `loop.varying` (false: a
/// fixed number of passes) last.
fn mainline_kvs(g: &Gguf, l: &Loop) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut n = 0u64;
    let prefix = format!("{ARCH}.");
    for kv in &g.kvs {
        let key = match kv.key.as_str() {
            "inkling2.num_loops" => format!("{LOOP_ARCH}.loop.count"),
            "inkling2.loop.core_start" => format!("{LOOP_ARCH}.loop.start"),
            "inkling2.loop.core_end" => format!("{LOOP_ARCH}.loop.end"),
            k => match k.strip_prefix(&prefix) {
                Some(rest) => format!("{LOOP_ARCH}.{rest}"),
                None => k.to_string(),
            },
        };
        let mut raw = kv.raw.clone();
        if kv.key == "general.architecture" {
            raw.clear();
            raw.extend_from_slice(&(LOOP_ARCH.len() as u64).to_le_bytes());
            raw.extend_from_slice(LOOP_ARCH.as_bytes());
        } else if kv.key == "inkling2.block_count" {
            raw = match kv.ty {
                T_U32 | T_I32 => (l.physical as u32).to_le_bytes().to_vec(),
                _ => l.physical.to_le_bytes().to_vec(),
            };
        } else if kv.key.starts_with(&prefix) && kv.ty == T_ARRAY {
            // A per-layer array counts every pass; mainline lists each
            // physical layer once, and the repeats share theirs.
            let at = u32::from_le_bytes(raw[..4].try_into()?);
            let len = u64::from_le_bytes(raw[4..12].try_into()?);
            let blocks = g.int("inkling2.block_count").unwrap_or(0);
            if len == blocks {
                let size = fixed_size(at).context("per-layer array of strings")?;
                let mut cut = raw[..4].to_vec();
                cut.extend_from_slice(&l.physical.to_le_bytes());
                cut.extend_from_slice(&raw[12..12 + (l.physical * size) as usize]);
                raw = cut;
            }
        }
        put_string(&mut out, &key);
        out.extend_from_slice(&kv.ty.to_le_bytes());
        out.extend_from_slice(&raw);
        n += 1;
    }
    put_string(&mut out, &format!("{LOOP_ARCH}.loop.varying"));
    out.extend_from_slice(&T_BOOL.to_le_bytes());
    out.push(0);
    n += 1;
    let mut head = n.to_le_bytes().to_vec();
    head.extend_from_slice(&out);
    Ok(head)
}

const DROPPED: [&str; 2] = ["loop_res_query.weight", "loop_kv_res_query.weight"];

/// Rewrite `path` in place into mainline's layout. `progress` is told the
/// share of the data copied so far. The new file is written beside the old
/// and put in its place once whole; nothing is left behind on failure.
pub fn rewrite(path: &Path, progress: impl Fn(f32)) -> Result<()> {
    let g = read(path)?;
    let l = plan(&g).context("not in a layout this can rewrite")?;
    let tmp: PathBuf = {
        let mut s = path.as_os_str().to_owned();
        s.push(".chaty-rewrite");
        s.into()
    };
    let result = write(&g, &l, path, &tmp, &progress);
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result?;
    std::fs::rename(&tmp, path).context("replacing the file with its rewrite")?;
    Ok(())
}

fn write(g: &Gguf, l: &Loop, path: &Path, tmp: &Path, progress: &impl Fn(f32)) -> Result<()> {
    let align = g.alignment;
    let spans = g.spans();
    // Kept tensors, in the order their data lies, packed as they were.
    let mut kept: Vec<(usize, u64, u64)> = Vec::new(); // (tensor, span, new offset)
    let mut at = 0u64;
    for &(i, span) in &spans {
        if DROPPED.contains(&g.tensors[i].name.as_str()) {
            continue;
        }
        kept.push((i, span, at));
        at += span;
    }
    // The message queries after them, one per looped layer: its q, k, v and
    // ffn rows — the attention and ffn queries from loop_res_query, the k and
    // v ones from loop_kv_res_query, two rows of each per layer.
    let mut src = File::open(path)?;
    let res = g.tensor(DROPPED[0]).context("loop_res_query")?;
    let kvq = g.tensor(DROPPED[1]).context("loop_kv_res_query")?;
    let core = l.end - l.start;
    let mut read_rows = |t: &TensorInfo| -> Result<Vec<u8>> {
        let mut b = vec![0u8; (l.row * 2 * core) as usize];
        src.seek(SeekFrom::Start(g.data_start + t.offset))?;
        src.read_exact(&mut b)?;
        Ok(b)
    };
    let (res_b, kv_b) = (read_rows(res)?, read_rows(kvq)?);
    let row = |b: &[u8], k: u64| b[(k * l.row) as usize..((k + 1) * l.row) as usize].to_vec();
    let mut queries = Vec::new();
    let mut q_at = at.div_ceil(align) * align;
    for j in 0..core {
        let mut data = row(&res_b, 2 * j);
        data.extend(row(&kv_b, 2 * j));
        data.extend(row(&kv_b, 2 * j + 1));
        data.extend(row(&res_b, 2 * j + 1));
        queries.push((format!("blk.{}.loop_msg_query.weight", l.start + j), q_at, data));
        q_at += (4 * l.row).div_ceil(align) * align;
    }

    // Header: the metadata, then the tensors in their original order with
    // their new names and offsets, then the queries.
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&g.version.to_le_bytes());
    head.extend_from_slice(&((kept.len() + queries.len()) as u64).to_le_bytes());
    head.extend(mainline_kvs(g, l)?);
    let new_offset: std::collections::HashMap<usize, u64> = kept.iter().map(|&(i, _, o)| (i, o)).collect();
    for (i, t) in g.tensors.iter().enumerate() {
        let Some(&offset) = new_offset.get(&i) else { continue };
        put_string(&mut head, &mainline_name(&t.name));
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.ty.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
    }
    for (name, offset, _) in &queries {
        put_string(&mut head, name);
        head.extend_from_slice(&2u32.to_le_bytes());
        head.extend_from_slice(&res.dims[0].to_le_bytes());
        head.extend_from_slice(&4u64.to_le_bytes());
        head.extend_from_slice(&res.ty.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
    }
    let pad = (head.len() as u64).div_ceil(align) * align - head.len() as u64;
    head.extend(std::iter::repeat_n(0u8, pad as usize));

    let total = head.len() as u64 + q_at;
    if let Some(free) = free_space(tmp) {
        if free < total + (256 << 20) {
            bail!(trf!(
                "磁盘空间不足:改写这个模型需要 {:.1} GB 可用空间,现在只有 {:.1} GB。",
                "not enough disk space: rewriting this model needs {:.1} GB free, and {:.1} GB is.",
                total as f64 / 1e9,
                free as f64 / 1e9
            ));
        }
    }

    let mut out = BufWriter::with_capacity(8 << 20, File::create(tmp)?);
    out.write_all(&head)?;
    let mut buf = vec![0u8; 16 << 20];
    let mut done = 0u64;
    let mut told = 0f32;
    for &(i, span, _) in &kept {
        src.seek(SeekFrom::Start(g.data_start + g.tensors[i].offset))?;
        let mut left = span;
        while left > 0 {
            let n = left.min(buf.len() as u64) as usize;
            src.read_exact(&mut buf[..n])?;
            out.write_all(&buf[..n])?;
            left -= n as u64;
            done += n as u64;
            let frac = done as f32 / q_at.max(1) as f32;
            if frac - told >= 0.005 {
                told = frac;
                progress(frac);
            }
            if super::load_cancelled() {
                bail!("LOAD_CANCELLED");
            }
        }
    }
    let mut written = at;
    for (_, offset, data) in &queries {
        out.write_all(&vec![0u8; (offset - written) as usize])?;
        out.write_all(data)?;
        written = offset + data.len() as u64;
    }
    let file = out.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    progress(1.0);
    Ok(())
}

/// Bytes free on the volume that holds `path`.
fn free_space(path: &Path) -> Option<u64> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let dir = path.parent()?;
    disks
        .iter()
        .filter(|d| dir.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small file in Unsloth's looped layout: two physical layers before
    /// the loop, two in it, run three times — block_count 2 + 2×3 = 8.
    fn unsloth_file(dir: &Path) -> PathBuf {
        let mut kv = Vec::new();
        let mut n_kv = 0u64;
        let mut put = |key: &str, ty: u32, raw: Vec<u8>| {
            put_string(&mut kv, key);
            kv.extend_from_slice(&ty.to_le_bytes());
            kv.extend(raw);
            n_kv += 1;
        };
        let s = |v: &str| {
            let mut b = (v.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(v.as_bytes());
            b
        };
        put("general.architecture", T_STRING, s("inkling2"));
        put("inkling2.block_count", T_U32, 8u32.to_le_bytes().to_vec());
        let mut heads = T_U32.to_le_bytes().to_vec();
        heads.extend(8u64.to_le_bytes());
        for i in 0..8u32 {
            heads.extend((10 + i).to_le_bytes());
        }
        put("inkling2.attention.head_count", T_ARRAY, heads);
        put("inkling2.num_loops", T_U32, 3u32.to_le_bytes().to_vec());
        put("inkling2.loop.core_start", T_U32, 1u32.to_le_bytes().to_vec());
        put("inkling2.loop.core_end", T_U32, 3u32.to_le_bytes().to_vec());
        put("inkling2.loop.gate_rank", T_U32, 2u32.to_le_bytes().to_vec());
        put("tokenizer.ggml.pre", T_STRING, s("inkling2"));
        // Tensors: the two model-wide query tensors (4 rows of 2 f32 each),
        // one ordinary tensor and one loop control tensor.
        let embd = 2u64;
        let rows = |base: f32| (0..4).flat_map(|r| (0..embd).map(move |c| base + r as f32 * 10.0 + c as f32)).flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let tensors: Vec<(&str, Vec<u64>, Vec<u8>)> = vec![
            ("loop_kv_res_query.weight", vec![embd, 4], rows(100.0)),
            ("loop_res_query.weight", vec![embd, 4], rows(0.0)),
            ("token_embd.weight", vec![embd, 3], (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect()),
            ("blk.1.loop_mlp_gain.weight", vec![embd], vec![7u8; 8]),
        ];
        let mut head = b"GGUF".to_vec();
        head.extend(3u32.to_le_bytes());
        head.extend((tensors.len() as u64).to_le_bytes());
        head.extend(n_kv.to_le_bytes());
        head.extend(kv);
        let mut data = Vec::new();
        for (name, dims, bytes) in &tensors {
            put_string(&mut head, name);
            head.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                head.extend(d.to_le_bytes());
            }
            head.extend(0u32.to_le_bytes());
            head.extend((data.len() as u64).to_le_bytes());
            data.extend(bytes);
            while data.len() % 32 != 0 {
                data.push(0);
            }
        }
        // The source's last tensor carries no trailing padding.
        let last = tensors.last().unwrap().2.len();
        data.truncate(data.len() - (32 - last));
        while head.len() % 32 != 0 {
            head.push(0);
        }
        head.extend(data);
        let path = dir.join("model.gguf");
        std::fs::write(&path, head).unwrap();
        path
    }

    #[test]
    fn unsloths_loop_is_rewritten_as_mainline_reads_it() {
        let dir = std::env::temp_dir().join(format!("chaty-gguf-layout-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = unsloth_file(&dir);
        assert!(needs_rewrite(&path));
        rewrite(&path, |_| {}).unwrap();
        assert!(!needs_rewrite(&path), "done once");
        assert!(!dir.join("model.gguf.chaty-rewrite").exists());

        let g = read(&path).unwrap();
        assert_eq!(g.str("general.architecture").as_deref(), Some("inkling2_loop"));
        assert_eq!(g.int("inkling2_loop.block_count"), Some(4), "2 + 2 physical layers");
        assert_eq!(g.int("inkling2_loop.loop.count"), Some(3));
        assert_eq!(g.int("inkling2_loop.loop.start"), Some(1));
        assert_eq!(g.int("inkling2_loop.loop.end"), Some(3));
        assert_eq!(g.int("inkling2_loop.loop.gate_rank"), Some(2));
        assert_eq!(g.kvs.last().map(|k| (k.key.as_str(), k.raw.clone())), Some(("inkling2_loop.loop.varying", vec![0])));
        let heads = &g.kv("inkling2_loop.attention.head_count").unwrap().raw;
        assert_eq!(u64::from_le_bytes(heads[4..12].try_into().unwrap()), 4);
        assert_eq!(heads[12..].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), vec![10, 11, 12, 13]);
        assert_eq!(g.str("tokenizer.ggml.pre").as_deref(), Some("inkling2"), "not an architecture key");
        let names: Vec<&str> = g.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["token_embd.weight", "blk.1.loop_ffn_ctrl_gain.weight", "blk.1.loop_msg_query.weight", "blk.2.loop_msg_query.weight"]);

        let file = std::fs::read(&path).unwrap();
        let at = |name: &str| {
            let t = g.tensor(name).unwrap();
            (g.data_start + t.offset) as usize
        };
        let f32s = |from: usize, n: usize| file[from..from + 4 * n].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
        assert_eq!(f32s(at("token_embd.weight"), 6), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&file[at("blk.1.loop_ffn_ctrl_gain.weight")..][..8], &[7u8; 8]);
        // Layer 1: res row 0, kv rows 0 and 1, res row 1. Layer 2: rows 2, 3.
        assert_eq!(f32s(at("blk.1.loop_msg_query.weight"), 8), vec![0.0, 1.0, 100.0, 101.0, 110.0, 111.0, 10.0, 11.0]);
        assert_eq!(f32s(at("blk.2.loop_msg_query.weight"), 8), vec![20.0, 21.0, 120.0, 121.0, 130.0, 131.0, 30.0, 31.0]);
        assert!(g.tensors.iter().all(|t| t.offset % 32 == 0));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_mainline_reads_is_left_alone() {
        let dir = std::env::temp_dir().join(format!("chaty-gguf-layout-alone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = unsloth_file(&dir);
        rewrite(&path, |_| {}).unwrap();
        let before = std::fs::read(&path).unwrap();
        assert!(rewrite(&path, |_| {}).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The real file: the rewrite of Unsloth's root Inkling-2-Mini GGUF is
    /// the one they publish for mainline, byte for byte.
    /// CHATY_LAYOUT_GGUF=<a copy of the root file> CHATY_LAYOUT_SHA=<sha256 of
    /// the Mainline_Support one> cargo test --release --lib gguf_layout -- --ignored
    #[test]
    #[ignore]
    fn the_rewrite_is_unsloths_mainline_file() {
        use sha2::Digest;
        let path = PathBuf::from(std::env::var("CHATY_LAYOUT_GGUF").expect("CHATY_LAYOUT_GGUF"));
        let want = std::env::var("CHATY_LAYOUT_SHA").expect("CHATY_LAYOUT_SHA");
        let t0 = std::time::Instant::now();
        rewrite(&path, |_| {}).unwrap();
        eprintln!("rewritten in {:.1} s", t0.elapsed().as_secs_f32());
        let mut f = File::open(&path).unwrap();
        let mut h = sha2::Sha256::new();
        std::io::copy(&mut f, &mut h).unwrap();
        assert_eq!(format!("{:x}", h.finalize()), want);
    }
}
