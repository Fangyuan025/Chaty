"""Stage 1 — synthesize the voiceover per scene with edge-tts and record
word-level timestamps used later for karaoke captions.

Usage: python scripts/tts.py <project_dir>
Writes: audio/scene_NN.mp3 and audio/timing.json
"""
import asyncio
import json
import sys

from common import (audio_duration, clean_for_tts, die, ensure_dirs, load_storyboard, log,
                    project_paths)


async def synth_scene(text: str, voice: str, rate: str, out_path) -> list:
    import edge_tts

    words = []
    cm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    with open(out_path, "wb") as f:
        async for chunk in cm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({
                    "w": chunk["text"],
                    "t0": round(chunk["offset"] / 1e7, 3),
                    "t1": round((chunk["offset"] + chunk["duration"]) / 1e7, 3),
                })
    return words


async def synth_with_retry(text, voice, rate, out_path, attempts=4):
    last = None
    for i in range(attempts):
        try:
            words = await synth_scene(text, voice, rate, out_path)
            if words and out_path.stat().st_size > 1024:
                return words
            last = RuntimeError("empty synthesis result")
        except Exception as e:  # noqa: BLE001 - network flake, retry
            last = e
        await asyncio.sleep(1.5 * (i + 1))
    raise RuntimeError(f"edge-tts failed after {attempts} attempts: {last}")


async def main(project_dir):
    sb = load_storyboard(project_dir)
    paths = project_paths(project_dir)
    ensure_dirs(paths)

    scenes_out = []
    for i, sc in enumerate(sb["scenes"], 1):
        text = clean_for_tts(sc["text"])
        out = paths["audio"] / f"scene_{i:02d}.mp3"
        log(f"[tts] scene {i}/{len(sb['scenes'])}: {text[:36]}...")
        words = await synth_with_retry(text, sb["voice"], sb["rate"], out)
        dur = audio_duration(out)
        vo_end = max(w["t1"] for w in words)
        scenes_out.append({
            "i": i,
            "file": out.name,
            "audio_dur": round(dur, 3),
            "vo_end": round(vo_end, 3),
            "words": words,
        })

    paths["timing"].write_text(json.dumps({"scenes": scenes_out}, ensure_ascii=False, indent=1))
    total = sum(s["vo_end"] + 0.35 for s in scenes_out)
    log(f"[tts] done. {len(scenes_out)} scenes, estimated video length ~{total:.1f}s")
    if total > 90:
        log("[tts] WARNING: over 90s — consider trimming the script for short-form platforms")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        die("usage: python scripts/tts.py <project_dir>")
    asyncio.run(main(sys.argv[1]))
