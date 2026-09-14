#!/bin/bash
set -e
python3 - <<'PY'
import re
s = open("index.html").read()
assert "--accent: #2c6fb5;" in s and "--paper: #eef3f7;" in s, "theme"
assert "card-note" not in s and s.count("margin-note") == 5, f"rename: {s.count('margin-note')}"
i6 = s.index('id="s6"'); ix = s.index('id="s6b"'); i7 = s.index('id="s7"')
assert i6 < ix < i7 and "Chapter 6½" in s and "Extra notes." in s, "new section"
assert len(re.findall(r'<section class="panel-\d+"', s)) == 12, "sections intact"
PY
node -e '
  const s = require("fs").readFileSync("index.html", "utf8");
  const code = s.slice(s.indexOf("function bumpVisits"), s.indexOf("function renderVisits"));
  const bumpVisits = new Function(code + "; return bumpVisits;")();
  const m = new Map(); const store = { getItem: k => m.get(k) ?? null, setItem: (k, v) => m.set(k, v) };
  const a = bumpVisits(store), b = bumpVisits(store);
  if (!(b === a + 1 && Number(m.get("visits")) === b)) { console.error("counter", a, b, m.get("visits")); process.exit(1); }
'
