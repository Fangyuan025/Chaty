"""Stage 4 — compose the final video with core ffmpeg filters only.

Per scene: media -> cover-crop -> Ken Burns motion (images) -> caption PNG
overlays (Pillow-rendered) -> h264 intermediate. Then concat, voiceover
assembly, BGM with sidechain ducking, and EBU loudness normalization.

Usage: python scripts/compose.py <project_dir>
Writes: final.mp4
"""
import json
import math
import sys
from pathlib import Path

from captions import (BadgeRenderer, CaptionRenderer, HookRenderer, StickyRenderer,
                      group_words)
from common import (ASPECTS, die, ensure_dirs, load_storyboard, log, media_info, project_paths,
                    run)

PAD_AFTER = 0.12          # breathing room after each scene's VO
TAIL_EXTRA = 0.35         # extra hold on the last scene
VO_TAIL = 0.24            # keep this much silence after the last word of a scene


def snap(t, fps):
    return round(t * fps) / fps


def kb_expr(effect, frames):
    """zoompan z/x/y expressions for a given Ken Burns effect."""
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    d = max(frames, 1)
    if effect == "kb_in":
        return f"z='1+0.13*on/{d}':x='{cx}':y='{cy}'"
    if effect == "kb_out":
        return f"z='1.13-0.13*on/{d}':x='{cx}':y='{cy}'"
    if effect == "pan_left":
        return f"z='1.10':x='(iw-iw/zoom)*(1-on/{d})':y='(ih-ih/zoom)/2'"
    if effect == "pan_right":
        return f"z='1.10':x='(iw-iw/zoom)*on/{d}':y='(ih-ih/zoom)/2'"
    return f"z='1.02':x='{cx}':y='{cy}'"  # static


def cover_dims(w, h, tw, th):
    f = max(tw / w, th / h)
    return math.ceil(w * f / 2) * 2, math.ceil(h * f / 2) * 2


EFFECT_CYCLE = ["kb_in", "kb_out", "pan_right", "pan_left"]


def build_scene_captions(sb, scene, timing, renderer, workdir, si):
    """Render caption PNGs for one scene.
    Returns list of overlays: {file, t0, t1, fade}."""
    style = sb["caption_style"]
    if style == "none":
        return []
    lines = group_words(scene["text"], timing["words"], sb["lang"])
    overlays = []
    for li, line in enumerate(lines):
        if style == "pop":
            png = workdir / f"cap_s{si:02d}_{li:02d}.png"
            renderer.render(renderer.line_pieces_pop(line, scene.get("emphasis")), png)
            overlays.append({"file": png, "t0": line["t0"], "t1": line["t1"], "fade": True})
        else:  # karaoke word states
            bounds = [line["t0"]] + [w["t0"] for w in line["words"][1:]] + [line["t1"]]
            for j in range(len(line["words"])):
                png = workdir / f"cap_s{si:02d}_{li:02d}_w{j:02d}.png"
                renderer.render(renderer.line_pieces_karaoke(line, j), png)
                t0, t1 = bounds[j], bounds[j + 1]
                if t1 - t0 < 0.01:
                    continue
                overlays.append({"file": png, "t0": t0, "t1": t1, "fade": False})
    return overlays


def split_frames(frames, n):
    base, rem = divmod(frames, n)
    return [base + (1 if k < rem else 0) for k in range(n)]


def shot_effects(scene, si, n):
    """Effect per shot: scene effect first (or alternating default), then cycle."""
    if scene["effect"] == "static":
        return ["static"] * n
    start = scene["effect"] if scene["effect"] != "auto" else \
        EFFECT_CYCLE[(si - 1) % 2]
    idx = EFFECT_CYCLE.index(start) if start in EFFECT_CYCLE else 0
    return [EFFECT_CYCLE[(idx + k) % len(EFFECT_CYCLE)] for k in range(n)]


def render_scene(sb, si, scene, timing, shots, paths, renderer, hook_png=None,
                 sticky_png=None, badge_png=None):
    W, H = ASPECTS[sb["aspect"]]
    fps = sb["fps"]
    vo_trim = min(timing["audio_dur"], timing["vo_end"] + VO_TAIL)
    scene_dur = snap(vo_trim + PAD_AFTER + (TAIL_EXTRA if si == len(sb["scenes"]) else 0), fps)
    frames = round(scene_dur * fps)
    out = paths["work"] / f"scene_{si:02d}.mp4"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    fc = []

    # ---- shot chain (multiple visuals per scene, hard cuts every ~3s) ----
    n = len(shots)
    fsplit = split_frames(frames, n)
    effects = shot_effects(scene, si, n)
    for k, (shot, fk, eff) in enumerate(zip(shots, fsplit, effects)):
        media = paths["media"] / shot["file"]
        shot_dur = fk / fps
        if shot["kind"] == "image":
            cw, ch = cover_dims(shot["w"], shot["h"], int(W * 1.3), int(H * 1.3))
            rw, rh = int(W * 1.5) // 2 * 2, int(H * 1.5) // 2 * 2
            cmd += ["-i", media]
            fc.append(
                f"[{k}:v]scale={cw}:{ch},crop={int(W*1.3)//2*2}:{int(H*1.3)//2*2},"
                f"zoompan={kb_expr(eff, fk)}:d={fk}:s={rw}x{rh}:fps={fps},"
                f"scale={W}:{H}:flags=lanczos,setsar=1,format=yuv420p[sh{k}]")
        else:
            vdur = shot["dur"]
            if vdur and vdur > shot_dur * 1.7:
                cmd += ["-ss", f"{(vdur - shot_dur) / 2:.2f}"]
            if vdur and vdur < shot_dur:
                cmd += ["-stream_loop", "-1"]
            cmd += ["-i", media]
            cw, ch = cover_dims(shot["w"], shot["h"], W, H)
            fc.append(f"[{k}:v]scale={cw}:{ch},crop={W}:{H},fps={fps},"
                      f"trim=duration={shot_dur:.4f},setpts=PTS-STARTPTS,"
                      f"setsar=1,format=yuv420p[sh{k}]")
    if n > 1:
        fc.append("".join(f"[sh{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[cat]")
    else:
        fc.append("[sh0]null[cat]")
    # subtle color pop, marketing look
    fc.append("[cat]eq=contrast=1.05:saturation=1.13[base]")
    prev = "base"
    nin = n  # next ffmpeg input index

    # ---- white flash transition at scene start (except first scene) ----
    if si > 1:
        cmd += ["-f", "lavfi", "-i", f"color=white:s={W}x{H}:r={fps}:d=0.14"]
        fc.append(f"[{nin}:v]format=rgba,fade=t=out:st=0:d=0.14:alpha=1[fl]")
        fc.append(f"[{prev}][fl]overlay=enable='between(t,0,0.14)'[vfl]")
        prev = "vfl"
        nin += 1

    # ---- overlays: sticky title, hook, badge, captions ----
    overlays = []
    if sticky_png:
        overlays.append({"file": sticky_png, "t0": 0.0, "t1": scene_dur + 1,
                         "fade": False, "y": int(H * 0.045)})
    if hook_png and si == 1:
        overlays.append({"file": hook_png, "t0": 0.0,
                         "t1": float(sb["hook"].get("seconds", 2.5)), "fade": True,
                         "y": int(H * 0.13)})
    if badge_png:
        overlays.append({"file": badge_png, "t0": 0.04, "t1": min(2.2, scene_dur),
                         "fade": True, "y": int(H * 0.26)})
    overlays += build_scene_captions(sb, scene, timing, renderer, paths["work"], si)

    for ov in overlays:
        if ov["fade"]:
            cmd += ["-framerate", str(fps), "-loop", "1"]
        cmd += ["-i", ov["file"]]
        chain = "format=rgba"
        if ov["fade"]:
            chain += f",fade=t=in:st={ov['t0']:.3f}:d=0.09:alpha=1"
        fc.append(f"[{nin}:v]{chain}[c{nin}]")
        y = ov.get("y")
        ypos = y if y is not None else f"{int(H * sb['caption']['position'])}-h/2"
        fc.append(f"[{prev}][c{nin}]overlay=x=(W-w)/2:y={ypos}"
                  f":enable='between(t,{ov['t0']:.3f},{ov['t1']:.3f})'[v{nin}]")
        prev = f"v{nin}"
        nin += 1

    cmd += ["-filter_complex", ";".join(fc), "-map", f"[{prev}]",
            "-t", f"{scene_dur:.4f}", "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", out]
    run(cmd)
    return out, scene_dur, vo_trim


def build_voiceover(paths, scene_specs):
    """Concat per-scene VO (trimmed + padded to scene length) into work/vo.wav."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    fc = []
    for k, spec in enumerate(scene_specs):
        cmd += ["-i", paths["audio"] / spec["file"]]
        fc.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,"
                  f"atrim=0:{spec['vo_trim']:.4f},asetpts=PTS-STARTPTS,"
                  f"apad=whole_dur={spec['scene_dur']:.4f}[a{k}]")
    fc.append("".join(f"[a{k}]" for k in range(len(scene_specs))) +
              f"concat=n={len(scene_specs)}:v=0:a=1[vo]")
    out = paths["work"] / "vo.wav"
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vo]", "-c:a", "pcm_s16le", out]
    run(cmd)
    return out


def make_whoosh(paths):
    """Synthesize a short transition whoosh (no external SFX assets needed)."""
    out = paths["work"] / "whoosh.wav"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "anoisesrc=d=0.5:color=pink:amplitude=0.55:seed=42",
         "-af", "highpass=f=300,lowpass=f=2600,"
                "afade=t=in:st=0:d=0.22:curve=esin,afade=t=out:st=0.22:d=0.28:curve=esin,"
                "aresample=48000,aformat=channel_layouts=stereo",
         out])
    return out


def final_mux(paths, sb, scene_specs):
    total_dur = sum(s["scene_dur"] for s in scene_specs)
    video = paths["work"] / "video_full.mp4"
    vo = paths["work"] / "vo.wav"
    bgm = next(paths["root"].glob("bgm.*"), None)
    wants_bgm = sb["bgm"].get("mood") != "none" or sb["bgm"].get("file")
    if wants_bgm and bgm is None:
        die("storyboard wants BGM but no bgm.* file in project — run scripts/bgm.py first "
            "(or set bgm mood to \"none\")")
    has_bgm = bgm is not None and wants_bgm

    # scene boundary times for transition whooshes
    cuts = []
    t = 0.0
    for s in scene_specs[:-1]:
        t += s["scene_dur"]
        cuts.append(max(0.0, t - 0.22))
    has_sfx = sb.get("sfx", True) and cuts

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", vo]
    fc = []
    mix_inputs = ["[1:a]"]
    nin = 2
    if has_bgm:
        gain = sb["bgm"].get("gain_db", -16)
        fade_out = max(total_dur - 1.6, 0)
        cmd += ["-stream_loop", "-1", "-i", bgm]
        fc.append(f"[{nin}:a]aresample=48000,aformat=channel_layouts=stereo,volume={gain}dB,"
                  f"atrim=0:{total_dur:.4f},asetpts=PTS-STARTPTS,"
                  f"afade=t=in:d=0.8,afade=t=out:st={fade_out:.3f}:d=1.6[bg]")
        fc.append(f"[bg][1:a]sidechaincompress=threshold=0.03:ratio=12:attack=15:release=350[bgd]")
        mix_inputs.append("[bgd]")
        nin += 1
    if has_sfx:
        whoosh = make_whoosh(paths)
        cmd += ["-i", whoosh]
        fc.append(f"[{nin}:a]asplit={len(cuts)}" +
                  "".join(f"[w{k}]" for k in range(len(cuts))))
        for k, ct in enumerate(cuts):
            ms = int(ct * 1000)
            fc.append(f"[w{k}]adelay={ms}|{ms},volume=-13dB[wd{k}]")
        fc.append("".join(f"[wd{k}]" for k in range(len(cuts))) +
                  f"amix=inputs={len(cuts)}:normalize=0,apad=whole_dur={total_dur:.4f}[sfx]")
        mix_inputs.append("[sfx]")
        nin += 1

    if len(mix_inputs) > 1:
        fc.append("".join(mix_inputs) +
                  f"amix=inputs={len(mix_inputs)}:duration=first:normalize=0,"
                  f"loudnorm=I=-15:TP=-1.5:LRA=11[aout]")
    else:
        fc.append("[1:a]loudnorm=I=-15:TP=-1.5:LRA=11[aout]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", paths["final"]]
    run(cmd)


def main(project_dir):
    sb = load_storyboard(project_dir)
    paths = project_paths(project_dir)
    ensure_dirs(paths)
    if not paths["timing"].exists():
        die("audio/timing.json missing — run scripts/tts.py first")
    if not paths["manifest"].exists():
        die("media/manifest.json missing — run scripts/assets.py first")
    timing = {s["i"]: s for s in json.loads(paths["timing"].read_text())["scenes"]}
    manifest = {}
    for m in json.loads(paths["manifest"].read_text()):
        manifest[m["i"]] = m.get("shots") or [{k: v for k, v in m.items() if k != "i"}]

    W, H = ASPECTS[sb["aspect"]]
    renderer = CaptionRenderer(W, H, sb["lang"], highlight=sb["caption"]["highlight"],
                               uppercase=sb["caption"]["uppercase"])
    highlight = sb["caption"]["highlight"]
    hook_png = None
    if sb.get("hook"):
        hook_png = paths["work"] / "hook.png"
        hr = HookRenderer(W, H, sb["lang"], highlight=highlight)
        hr.render([(sb["hook"]["text"], hr.highlight)], hook_png)
    sticky_png = None
    if sb.get("sticky_title"):
        sticky_png = paths["work"] / "sticky.png"
        StickyRenderer(W, H, sb["lang"], highlight=highlight).render_sticky(
            sb["sticky_title"]["text"], sticky_png)
    badge_renderer = BadgeRenderer(W, H, sb["lang"], highlight=highlight)

    scene_specs = []
    for si in range(1, len(sb["scenes"]) + 1):
        if si not in timing or si not in manifest:
            die(f"scene {si}: missing timing or asset — rerun earlier stages")
        scene = sb["scenes"][si - 1]
        badge_png = None
        if scene.get("badge"):
            badge_png = paths["work"] / f"badge_{si:02d}.png"
            badge_renderer.render_badge(scene["badge"], badge_png)
        log(f"[compose] scene {si}/{len(sb['scenes'])} "
            f"({len(manifest[si])} shots, {sb['caption_style']} captions)")
        out, scene_dur, vo_trim = render_scene(
            sb, si, scene, timing[si], manifest[si], paths, renderer,
            hook_png=hook_png, sticky_png=sticky_png, badge_png=badge_png)
        scene_specs.append({"file": timing[si]["file"], "scene_dur": scene_dur,
                            "vo_trim": vo_trim, "mp4": out})

    concat_txt = paths["work"] / "concat.txt"
    concat_txt.write_text("".join(f"file '{s['mp4'].name}'\n" for s in scene_specs))
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", concat_txt, "-c", "copy", paths["work"] / "video_full.mp4"])

    log("[compose] assembling voiceover + bgm + sfx + loudness…")
    build_voiceover(paths, scene_specs)
    total = sum(s["scene_dur"] for s in scene_specs)
    final_mux(paths, sb, scene_specs)
    log(f"[compose] final.mp4 done — {total:.1f}s. Now run scripts/check.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        die("usage: python scripts/compose.py <project_dir>")
    main(sys.argv[1])
