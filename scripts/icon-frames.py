#!/usr/bin/env python3
"""The small frames of icons/icon.ico (and icons/32x32.png), drawn for their size.

A frame scaled down from the 512 px icon turns its strokes to a grey haze: the
bubble's outline is 15 px at 512, half a pixel at 16. Windows shows these
frames on the taskbar, in the title bar, Alt-Tab and Explorer, so each one up
to 48 px is drawn again here from the same artwork, weighted for its size:

- the tile's edges on whole pixels, its corners the 512's proportion;
- the bubble is the 512's own outline, thickened evenly about its centre line;
- the mark inside is the 512's, rebuilt from its six points: its lines at a
  weight that reads, its dots from 30 px up (below that they run together);
- the white is moved by up to half a pixel to where it lands crispest.

64 and 256 are the 512 scaled down, as before. The 32 px frame comes first:
Tauri makes the window's icon from the first frame (see src/winicon.rs for how
Windows then gets every size from the executable instead).

Needs numpy, scipy and Pillow:  python3 scripts/icon-frames.py
"""
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
ICONS = ROOT / "src-tauri" / "icons"
MASTER = ICONS / "icon.png"

BG = np.array([48, 63, 82], float)  # the tile's slate, sampled from the 512
# The tile in the 512: x and y from 16 to 493 (inclusive).
TILE0, TILE1 = 16.0, 494.0
TILE = TILE1 - TILE0
CORNER = 0.19  # corner radius, as a share of the tile's side
BUBBLE_W = 15.0  # the bubble's stroke in the 512
# The mark: its six points in the 512 (pixel centres, hence the half), and the
# lines between them.
DOTS = [(x + 0.5, y + 0.5) for x, y in [(196.6, 260.4), (221.6, 211.5), (249.1, 160.5), (278.5, 211.6), (312.7, 160.7), (312.8, 260.2)]]
LINES = [(0, 1), (1, 2), (2, 3), (3, 5), (4, 5)]
DOT_R = 13.0

# Per frame: tile side in px, bubble stroke, mark line, dot radius (0 = none), in output px.
FRAMES = {
    16: (16, 1.25, 1.0, 0.0),
    20: (20, 1.35, 1.1, 0.0),
    24: (22, 1.5, 1.15, 0.0),
    30: (28, 1.7, 1.25, 1.1),
    32: (30, 1.75, 1.25, 1.15),
    36: (34, 1.9, 1.35, 1.3),
    40: (38, 2.0, 1.4, 1.4),
    48: (46, 2.25, 1.6, 1.7),
}
SS = 48  # supersampling per output pixel


_BUBBLE = None


def master_bubble():
    """The 512's bubble outline as a coverage mask, the mark taken out."""
    global _BUBBLE
    if _BUBBLE is not None:
        return _BUBBLE
    im = np.asarray(Image.open(MASTER).convert("RGBA")).astype(float)
    white = (im[..., :3].min(axis=2) > 180) & (im[..., 3] > 200)
    lab, n = ndi.label(white)
    sizes = ndi.sum(white, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    keep = ndi.binary_dilation(keep, iterations=2)
    # Coverage from brightness over the slate, so its edges stay soft.
    lum = im[..., :3].mean(axis=2)
    cov = np.clip((lum - BG.mean()) / (255 - BG.mean()), 0, 1)
    _BUBBLE = np.where(keep, cov, 0.0)
    return _BUBBLE


def to_grid(u, v, tile_px, off, size):
    """512 coordinates → high-res grid coordinates for a frame."""
    s = tile_px / TILE * SS
    o = ((size - tile_px) / 2) * SS
    return (u - TILE0) * s + o + off[0], (v - TILE0) * s + o + off[1]


def render(size, shift=(0.0, 0.0)):
    tile_px, bw, lw, dr = FRAMES[size]
    H = size * SS
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32) + 0.5
    # Tile: rounded square, edges on whole pixels.
    m = (size - tile_px) / 2 * SS
    side = tile_px * SS
    r = CORNER * side
    cx = np.clip(xx, m + r, m + side - r)
    cy = np.clip(yy, m + r, m + side - r)
    tile = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & (xx >= m) & (xx <= m + side) & (yy >= m) & (yy <= m + side)
    # Bubble: the 512's outline resampled onto the grid, then thickened.
    s = tile_px / TILE * SS
    o = m
    src = master_bubble()
    # Inverse map grid → 512 for an affine resample.
    inv = 1.0 / s
    shift_px = (shift[0] * SS, shift[1] * SS)
    bub = ndi.affine_transform(
        src,
        np.diag([inv, inv]),
        # grid pixel j (centre j + 0.5) → 512 coordinate → 512 pixel index (centre at +0.5)
        offset=[(0.5 - o - shift_px[1]) * inv + TILE0 - 0.5, (0.5 - o - shift_px[0]) * inv + TILE0 - 0.5],
        output_shape=(H, H),
        order=1,
    ) > 0.5
    master_w = BUBBLE_W * s  # in grid px
    grow = max(0.0, (bw * SS - master_w) / 2)
    if grow > 0:
        bub = ndi.distance_transform_edt(~bub) <= grow
    # Mark: lines and dots, as distances.
    pts = [to_grid(x, y, tile_px, shift_px, size) for x, y in DOTS]
    mark = np.zeros((H, H), bool)
    half = lw * SS / 2
    for a, b in LINES:
        (x0, y0), (x1, y1) = pts[a], pts[b]
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        mark |= (xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2 <= half * half
    if dr > 0:
        rr = dr * SS
        for x, y in pts:
            mark |= (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
    white = (bub | mark) & tile

    def down(a):
        return a.reshape(size, SS, size, SS).mean(axis=(1, 3))

    alpha = down(tile.astype(np.float32))
    wcov = down(white.astype(np.float32))
    # Colour of the covered part: white where the mark is, slate elsewhere.
    with np.errstate(invalid="ignore", divide="ignore"):
        wfrac = np.where(alpha > 0, wcov / alpha, 0)
    rgb = BG[None, None, :] * (1 - wfrac[..., None]) + 255.0 * wfrac[..., None]
    out = np.dstack([rgb, alpha[..., None] * 255]).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(out), wcov


def haze(wcov):
    """How much of the white sits on part-covered pixels: lower is crisper."""
    return float(np.minimum(wcov, 1 - wcov).sum())


def best_frame(size):
    steps = [i / 4 for i in range(-2, 2)]
    best = None
    for dx in steps:
        for dy in steps:
            img, w = render(size, (dx, dy))
            h = haze(w)
            if best is None or h < best[0]:
                best = (h, img, (dx, dy))
    return best


def ico(frames):
    """An ICO of PNG frames, in the order given."""
    head = struct.pack("<HHH", 0, 1, len(frames))
    dirs, blobs = b"", b""
    offset = 6 + 16 * len(frames)
    for im in frames:
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        data = buf.getvalue()
        w = im.width if im.width < 256 else 0
        dirs += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    return head + dirs + blobs


def main():
    master = Image.open(MASTER).convert("RGBA")
    drawn = {}
    for size in FRAMES:
        h, img, shift = best_frame(size)
        drawn[size] = img
        print(f"{size:>3} px  shift {shift}  haze {h:.1f}")
    big = {n: master.resize((n, n), Image.LANCZOS) for n in (64, 256)}
    order = [32, 16, 20, 24, 30, 36, 40, 48]
    frames = [drawn[n] for n in order] + [big[64], big[256]]
    (ICONS / "icon.ico").write_bytes(ico(frames))
    drawn[32].save(ICONS / "32x32.png", optimize=True)
    print("wrote icon.ico (" + ", ".join(str(f.width) for f in frames) + ") and 32x32.png")


if __name__ == "__main__":
    main()
