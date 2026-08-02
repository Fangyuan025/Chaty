---
name: cardlet
description: Turn text into polished carousel card images (PNG) via the Cardlet API
when: the user wants social/carousel cards — 小红书图文卡片, Instagram carousel, shareable card decks — from text or notes
---

Cardlet (https://usecardlet.vercel.app) renders styled carousel cards from
markdown-lite text over plain HTTP — bash + curl only, no browser needed.

1. **Write the deck** to `deck.json` in the workspace. `---` splits cards;
   one idea per card (heading + 1–3 short lines); EN/中文 both render well.

```json
{"deck":{"t":"Title ≤64","b":"Intro\n---\n## Point one\nDetail line.\n---\n%% 87% | a big stat card","a":"@handle","tp":"ink","s":"ig45"}}
```

   Keys: `t` title · `b` body (`## heading`, `- bullet`, `> quote`,
   `**bold**`, `==highlight==`, `%% stat | caption`, `---` new card) ·
   `a` author · `tp` template (free: `ink` `cloud` `noir` `peach`) ·
   `s` size (`ig45` 1080×1350, `xhs34` 小红书 1242×1656, `sq11` square,
   `story916`) · optional: `f` text size 0–4, `l` spacing 0–2, `bf`
   `sans`/`serif`, `c` cover card 0/1, `pn` page numbers 1/0.

2. **Render and save** (first call may take 10–20 s — cold start; retry once
   on failure):

```bash
curl -s --max-time 180 -X POST https://usecardlet.vercel.app/api/render \
  -H "Content-Type: application/json" -d @deck.json \
| python3 -c "import json,sys,base64; d=json.load(sys.stdin); assert 'images' in d, d; [open(f'card-{i+1:02d}.png','wb').write(base64.b64decode(p)) for i,p in enumerate(d['images'])]; print(d['pages'],'cards saved, watermark:',d['watermark'])"
```

3. **Report the saved PNG paths** to the user.

Limits: ≤12 cards per JSON call; for bigger decks fetch one card at a time —
`GET /api/render?d=p<base64url(deckJSON)>&page=N` returns a single binary PNG.
Be kind to the free API: sequential calls, no parallel bursts.

Pro: if the user gives a Cardlet Pro code (`CDLT.…`), add `"code":"CDLT…"`
beside `"deck"` — removes the footer watermark and unlocks pro templates
(`glass` `sunset` `mint` `press` `pop` `night` `terminal` `sakura` `gold`
`blueprint` `note`) plus `q` (QR sign-off URL). Without a code, cards carry a
small watermark — say so instead of working around it.
