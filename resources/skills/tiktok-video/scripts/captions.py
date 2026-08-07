"""Caption engine — groups TTS word timestamps into short punchy caption lines
and renders them to transparent PNGs with Pillow (no libass/drawtext needed).

Styles:
  karaoke — one PNG per word-state; the word being spoken is highlighted
  pop     — one PNG per line, faded in by ffmpeg; 'emphasis' substrings highlighted
"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import EMOJI_RE, ZH_PUNCT, emoji_font_path, main_font

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
SHADOW = (0, 0, 0, 150)


def hex_rgba(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


# ---------------------------------------------------------------- grouping

def group_words(text: str, words: list, lang: str, max_zh_chars=10, max_en_words=4):
    """Align TTS word tokens back onto the original text (which still has
    punctuation/emoji) and split them into short caption lines.

    Returns list of lines: {"words": [{"w","t0","t1","hard"}], "text": str}
    `hard` marks a token followed by sentence punctuation in the source text.
    """
    toks = []
    pos = 0
    lowered = text.lower()
    for w in words:
        token = w["w"]
        idx = lowered.find(token.lower(), pos)
        if idx < 0:
            # token not found verbatim (TTS normalization) — keep it, no punct info
            toks.append({**w, "hard": False})
            continue
        end = idx + len(token)
        rest = text[end:end + 3]
        nxt = rest.lstrip(" ")[:1]
        hard = bool(nxt) and nxt in tuple(ZH_PUNCT)
        toks.append({**w, "hard": hard, "sentence_end": nxt in "。!?!?…"})
        pos = end

    lines, cur = [], []

    def flush():
        if cur:
            lines.append({"words": list(cur), "text": "".join(t["w"] for t in cur) if lang == "zh"
                          else " ".join(t["w"] for t in cur)})
            cur.clear()

    if lang == "zh":
        for t in toks:
            chars = sum(len(x["w"]) for x in cur)
            if cur and chars + len(t["w"]) > max_zh_chars:
                flush()
            cur.append(t)
            if t.get("sentence_end") or (t["hard"] and sum(len(x["w"]) for x in cur) >= 4):
                flush()
    else:
        for t in toks:
            chars = sum(len(x["w"]) + 1 for x in cur)
            if cur and (len(cur) + 1 > max_en_words or chars + len(t["w"]) > 24):
                flush()
            cur.append(t)
            if t.get("sentence_end") or (t["hard"] and len(cur) >= 2):
                flush()
    flush()

    # display windows: line shows slightly early, holds until the next line
    for i, ln in enumerate(lines):
        ln["t0"] = max(0.0, ln["words"][0]["t0"] - 0.07)
        ln["t1"] = lines[i + 1]["words"][0]["t0"] - 0.07 if i + 1 < len(lines) \
            else ln["words"][-1]["t1"] + 0.30
    return lines


# ---------------------------------------------------------------- rendering

class CaptionRenderer:
    def __init__(self, W, H, lang, highlight="#FFE14D", uppercase=False):
        self.W, self.H, self.lang = W, H, lang
        self.highlight = hex_rgba(highlight)
        self.uppercase = uppercase
        self.size = int(W * (0.082 if lang == "zh" else 0.075))
        self.font = ImageFont.truetype(str(main_font(lang)), self.size)
        self.stroke = max(3, self.size // 9)
        ep = emoji_font_path()
        self.emoji_font = ImageFont.truetype(ep, 160) if ep else None

    # --- text measurement with emoji segments -------------------------
    def _segments(self, s):
        """Split string into [(kind, text)] where kind in {t, e}."""
        out, last = [], 0
        for m in EMOJI_RE.finditer(s):
            if m.start() > last:
                out.append(("t", s[last:m.start()]))
            out.append(("e", m.group()))
            last = m.end()
        if last < len(s):
            out.append(("t", s[last:]))
        return out

    def _seg_width(self, d, seg):
        kind, txt = seg
        if kind == "e":
            return int(self.size * 1.12) * len(EMOJI_RE.findall(txt)[0]) if self.emoji_font else 0
        return d.textlength(txt, font=self.font)

    def _draw_rich(self, img, d, x, y, pieces):
        """pieces: [(text, fill)] — draws with shadow + stroke; emoji via emoji font."""
        for txt, fill in pieces:
            for kind, seg in self._segments(txt):
                if kind == "e":
                    if not self.emoji_font:
                        continue
                    for ch in seg:
                        em = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
                        ImageDraw.Draw(em).text((20, 20), ch, font=self.emoji_font,
                                                embedded_color=True)
                        target = int(self.size * 1.12)
                        em = em.resize((int(200 * target / 160),) * 2, Image.LANCZOS)
                        img.alpha_composite(em, (int(x) - int(target * 0.12),
                                                 int(y) - int(target * 0.18)))
                        x += target
                else:
                    d.text((x + self.size * 0.045, y + self.size * 0.07), seg,
                           font=self.font, fill=SHADOW,
                           stroke_width=self.stroke, stroke_fill=SHADOW)
                    d.text((x, y), seg, font=self.font, fill=fill,
                           stroke_width=self.stroke, stroke_fill=BLACK)
                    x += d.textlength(seg, font=self.font)
        return x

    def _wrap_rows(self, pieces, max_w):
        """Split [(text, fill)] pieces into rows fitting max_w. Splits between
        words (en) or characters (zh)."""
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        rows, cur, cur_w = [], [], 0
        units = []
        for txt, fill in pieces:
            if self.lang == "en":
                for w in re.split(r"(\s+)", txt):
                    if w:
                        units.append((w, fill))
            else:
                for ch in txt:
                    units.append((ch, fill))
        for u_txt, u_fill in units:
            w = sum(self._seg_width(probe, s) for s in self._segments(u_txt))
            if cur and cur_w + w > max_w and u_txt.strip():
                rows.append(cur)
                cur, cur_w = [], 0
            if not cur and not u_txt.strip():
                continue
            cur.append((u_txt, u_fill))
            cur_w += w
        if cur:
            rows.append(cur)
        return rows

    def render(self, pieces, out_path: Path):
        """Render [(text, fill)] to a tight transparent PNG. Returns (w, h)."""
        max_w = int(self.W * 0.88)
        rows = self._wrap_rows(pieces, max_w)
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        row_h = int(self.size * 1.32)
        pad = self.stroke + int(self.size * 0.2)
        widths = [sum(self._seg_width(probe, s)
                      for t, _ in row for s in self._segments(t)) for row in rows]
        W = int(max(widths) + pad * 2) if widths else 10
        H = row_h * len(rows) + pad * 2
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for ri, row in enumerate(rows):
            x = (W - widths[ri]) / 2
            y = pad + ri * row_h
            self._draw_rich(img, d, x, y, row)
        img.save(out_path)
        return W, H

    # --- public: produce PNGs for a line ------------------------------
    def line_pieces_pop(self, line, emphasis):
        """Whole line; emphasis substrings tinted."""
        text = line["text"]
        if self.uppercase:
            text = text.upper()
        spans = []
        for em in emphasis or []:
            em2 = em.upper() if self.uppercase else em
            i = text.find(em2)
            if i >= 0:
                spans.append((i, i + len(em2)))
        spans.sort()
        pieces, pos = [], 0
        for a, b in spans:
            if a > pos:
                pieces.append((text[pos:a], WHITE))
            pieces.append((text[a:b], self.highlight))
            pos = b
        if pos < len(text):
            pieces.append((text[pos:], WHITE))
        return pieces or [(text, WHITE)]

    def line_pieces_karaoke(self, line, active_idx):
        pieces = []
        for j, w in enumerate(line["words"]):
            t = w["w"].upper() if self.uppercase else w["w"]
            if self.lang == "en" and j > 0:
                pieces.append((" ", WHITE))
            pieces.append((t, self.highlight if j == active_idx else WHITE))
        return pieces


class HookRenderer(CaptionRenderer):
    """Bigger top-of-video title card text."""

    def __init__(self, W, H, lang, highlight="#FFE14D"):
        super().__init__(W, H, lang, highlight=highlight)
        self.size = int(W * 0.105)
        self.font = ImageFont.truetype(str(main_font(lang)), self.size)
        self.stroke = max(4, self.size // 8)
