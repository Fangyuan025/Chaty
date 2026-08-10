// Source tooling for the Canvas code pane: element↔code correspondence and
// per-line syntax highlighting.
//
// `annotate` walks the raw HTML and tags every real element opening tag with
// `data-cv="N"`, remembering which SOURCE LINE each N starts on. The preview
// iframe renders the annotated HTML; an inspect shim reports the data-cv of
// whatever the user hovers/clicks, and the code pane maps it back to a line.
// The scan is context-aware: nothing inside <script>/<style> bodies or
// comments is touched (a `<div>` inside a JS string must not be rewritten).
import hljs from "highlight.js/lib/core";
import xml from "highlight.js/lib/languages/xml";
import javascript from "highlight.js/lib/languages/javascript";
import css from "highlight.js/lib/languages/css";

hljs.registerLanguage("xml", xml);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("css", css);

export interface AnnotatedHtml {
  /** The html with data-cv attributes injected. */
  html: string;
  /** cv index → 0-based source line of the element's opening tag. */
  lineOf: number[];
  /** cv index → tag name (for the hover chip). */
  tagOf: string[];
}

const VOIDISH = /^(!doctype|!--)/i;

export function annotate(source: string): AnnotatedHtml {
  const lineOf: number[] = [];
  const tagOf: string[] = [];
  let out = "";
  let i = 0;
  let line = 0;
  let ctx: "" | "script" | "style" | "comment" = "";
  const n = source.length;

  while (i < n) {
    const ch = source[i];
    if (ch === "\n") line++;

    if (ctx === "comment") {
      if (source.startsWith("-->", i)) {
        ctx = "";
        out += "-->";
        i += 3;
        continue;
      }
      out += ch;
      i++;
      continue;
    }
    if (ctx === "script" || ctx === "style") {
      const close = ctx === "script" ? "</script" : "</style";
      if (source.slice(i, i + close.length).toLowerCase() === close) {
        ctx = "";
        // fall through to normal tag handling for the closing tag
      } else {
        out += ch;
        i++;
        continue;
      }
    }

    if (ch === "<") {
      if (source.startsWith("<!--", i)) {
        ctx = "comment";
        out += "<!--";
        i += 4;
        continue;
      }
      const m = /^<([a-zA-Z][a-zA-Z0-9-]*)((?:"[^"]*"|'[^']*'|[^"'>])*)>/.exec(source.slice(i));
      if (m && !VOIDISH.test(m[1])) {
        const tag = m[1].toLowerCase();
        const attrs = m[2];
        const cv = lineOf.length;
        lineOf.push(line);
        tagOf.push(tag);
        const selfClose = attrs.endsWith("/");
        const body = (selfClose ? attrs.slice(0, -1) : attrs).trimEnd();
        out += `<${m[1]}${body} data-cv="${cv}"${selfClose ? " /" : ""}>`;
        line += (m[0].match(/\n/g) ?? []).length;
        i += m[0].length;
        if (tag === "script") ctx = "script";
        else if (tag === "style") ctx = "style";
        continue;
      }
    }

    out += ch;
    i++;
  }
  return { html: out, lineOf, tagOf };
}

/** Highlight a full HTML document and split into per-line markup, re-opening
 *  spans that hljs lets run across newlines so each line is self-contained. */
export function highlightLines(source: string): string[] {
  let value = "";
  try {
    value = hljs.highlight(source, { language: "xml" }).value;
  } catch {
    value = source.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  const lines: string[] = [];
  const open: string[] = []; // currently open span tags, verbatim
  let cur = "";
  let i = 0;
  const n = value.length;
  while (i < n) {
    const ch = value[i];
    if (ch === "\n") {
      lines.push(cur + "</span>".repeat(open.length));
      cur = open.join("");
      i++;
      continue;
    }
    if (ch === "<") {
      const close = value.startsWith("</span>", i);
      if (close) {
        open.pop();
        cur += "</span>";
        i += 7;
        continue;
      }
      const m = /^<span[^>]*>/.exec(value.slice(i));
      if (m) {
        open.push(m[0]);
        cur += m[0];
        i += m[0].length;
        continue;
      }
    }
    cur += ch;
    i++;
  }
  lines.push(cur + "</span>".repeat(open.length));
  return lines;
}

/**
 * Inspect shim injected into the preview iframe. Armed/disarmed by the parent
 * via postMessage. While armed: hovering outlines the element and reports its
 * data-cv up to the parent; clicking pins it. The parent can also ask the
 * shim to flash a specific data-cv (code line → element direction).
 */
export const INSPECT_SHIM = `<script>(function(){
  var armed = false, lastEl = null;
  var box = null;
  var selStyle = null;
  function ensureSelStyle(){
    if (selStyle) return;
    selStyle = document.createElement('style');
    selStyle.textContent = '.__cv_sel { outline: 2px solid #4c9ffe !important; outline-offset: 2px; }';
    document.documentElement.appendChild(selStyle);
  }
  function applySel(ids){
    ensureSelStyle();
    var old = document.querySelectorAll('.__cv_sel');
    for (var i = 0; i < old.length; i++) old[i].classList.remove('__cv_sel');
    (ids || []).forEach(function(id){
      var el = document.querySelector('[data-cv="' + id + '"]');
      if (el) el.classList.add('__cv_sel');
    });
  }
  function ensureBox(){
    if (box) return box;
    box = document.createElement('div');
    box.style.cssText = 'position:fixed;pointer-events:none;z-index:2147483647;border:2px solid #4c9ffe;background:rgba(76,159,254,0.12);border-radius:3px;transition:all 60ms linear;display:none';
    document.documentElement.appendChild(box);
    return box;
  }
  function showBox(el){
    var r = el.getBoundingClientRect();
    var b = ensureBox();
    b.style.display = 'block';
    b.style.left = r.left + 'px'; b.style.top = r.top + 'px';
    b.style.width = r.width + 'px'; b.style.height = r.height + 'px';
  }
  function cvOf(el){
    while (el && el.getAttribute && !el.getAttribute('data-cv')) el = el.parentElement;
    return el && el.getAttribute ? el.getAttribute('data-cv') : null;
  }
  document.addEventListener('mousemove', function(e){
    if (!armed) return;
    var cv = cvOf(e.target);
    if (cv === null) return;
    var el = document.querySelector('[data-cv="' + cv + '"]');
    if (el === lastEl) return;
    lastEl = el;
    if (el) showBox(el);
    try { parent.postMessage({ __chatyCvHover: cv }, '*'); } catch(_){}
  }, true);
  document.addEventListener('click', function(e){
    if (!armed) return;
    e.preventDefault(); e.stopPropagation();
    var cv = cvOf(e.target);
    if (cv !== null) {
      try { parent.postMessage({ __chatyCvSelect: { cv: cv, multi: !!(e.metaKey || e.ctrlKey) } }, '*'); } catch(_){}
    }
  }, true);
  window.addEventListener('message', function(e){
    var d = e.data || {};
    if (d.__chatyCvArm !== undefined) {
      armed = !!d.__chatyCvArm;
      if (!armed && box) box.style.display = 'none';
    }
    if (d.__chatyCvSetSel !== undefined) applySel(d.__chatyCvSetSel);
    if (d.__chatyCvFlash !== undefined) {
      var el = document.querySelector('[data-cv="' + d.__chatyCvFlash + '"]');
      if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); showBox(el);
        setTimeout(function(){ if (!armed && box) box.style.display = 'none'; }, 1600); }
    }
    // e2e hook: drive the exact hover/pick pipeline by selector — automation
    // drivers can't inject pointer events into a sandboxed srcdoc frame.
    if (d.__chatyCvSim) {
      var t = null;
      try { t = document.querySelector(d.__chatyCvSim.sel); } catch(_){}
      if (armed && t) {
        var cv = cvOf(t);
        if (cv !== null) {
          showBox(document.querySelector('[data-cv="' + cv + '"]') || t);
          try {
            parent.postMessage(
              d.__chatyCvSim.pick
                ? { __chatyCvSelect: { cv: cv, multi: !!d.__chatyCvSim.multi } }
                : { __chatyCvHover: cv },
              '*');
          } catch(_){}
        }
      }
    }
  });
})();</script>`;

/** Everything the model needs to fix the page in ONE round: the heal-banner
 *  error plus every error-level console line of the current version, deduped
 *  (same first-200-chars = same error re-thrown) and bounded. Numbered only
 *  when there are several, so the single-error payload stays byte-compatible
 *  with the old behavior. */
export function buildFixPayload(
  banner: string,
  consoleErrors: string[],
  lang: "zh" | "en" = "zh",
): string {
  // WebKit anonymizes sandboxed-frame errors to "Script error." — several
  // DISTINCT bugs collapse into identical, information-free lines (and then
  // into one after dedupe). Count them and say so, or the model fixes one
  // thing and honestly believes it fixed everything.
  const muzzled = [banner, ...consoleErrors].filter((t) => /^Script error\.?/.test(t.trim())).length;
  const seen = new Set<string>();
  const uniq: string[] = [];
  for (const t of [banner, ...consoleErrors]) {
    const trimmed = t.trim();
    if (!trimmed) continue;
    const key = trimmed.slice(0, 200);
    if (seen.has(key)) continue;
    seen.add(key);
    uniq.push(trimmed);
    if (uniq.length >= 12) break;
  }
  const note =
    muzzled > 0
      ? lang === "zh"
        ? `\n注意:其中 "Script error." 出现 ${muzzled} 次——这是浏览器把沙箱内的不同错误匿名化的结果,它们很可能是多个不同的运行时错误。定时器/事件监听里的错误现在自带行号,所以匿名条目通常来自脚本顶层直接执行的代码或内联 on*= 属性——重点排查这两处,不要只修一处。`
        : `\nNote: "Script error." appeared ${muzzled} time(s) — the browser anonymizes distinct sandboxed errors into this one line, so they are likely SEVERAL different runtime bugs. Timer/event-listener errors now carry line numbers, so anonymized entries usually come from TOP-LEVEL script statements or inline on*= attributes — audit those first; do not stop at one fix.`
      : "";
  if (uniq.length <= 1) return (uniq[0] ?? "") + note;
  return (uniq.map((t, i) => `${i + 1}. ${t}`).join("\n") + note).slice(0, 6400);
}

/** Parent-side syntax gate for the preview's inline scripts. WebKit muzzles
 *  every error inside a sandboxed null-origin srcdoc to "Script error." —
 *  but the Canvas HOLDS the source, so syntax errors can be recovered here
 *  in the parent context with real messages. new Function compiles without
 *  executing. (Function-body context: a stray top-level `return` slips
 *  through — acceptable for a diagnostic tier.) */
const CLASSIC_SCRIPT_TYPE = /^(text|application)\/(x-)?(java|ecma)script$/;

export function precheckScripts(html: string, lang: "zh" | "en" = "zh"): string[] {
  const out: string[] = [];
  let i = 0;
  for (const m of html.matchAll(/<script\b((?:[^>"']|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/script>/gi)) {
    const attrs = m[1];
    if (/\bsrc\s*=/i.test(attrs)) continue;
    i++;
    const body = m[2];
    if (!body.trim()) continue;
    // Only CLASSIC scripts compile under new Function. type="module"
    // (import/export), JSON/importmap data blocks and text/* templates are
    // legitimate pages — compiling them here produced FALSE syntax faults
    // (badge lit, bogus entry in the Fix payload). A module's real syntax
    // error still surfaces at runtime through the error shim's fault path.
    const tm = /\btype\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(attrs);
    const t = (tm?.[1] ?? tm?.[2] ?? tm?.[3] ?? "").trim().toLowerCase();
    if (t && !CLASSIC_SCRIPT_TYPE.test(t)) continue;
    try {
      new Function(body);
    } catch (e) {
      const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      out.push(
        lang === "zh"
          ? `语法预检(第 ${i} 个 script 块): ${msg}`
          : `Syntax precheck (script block ${i}): ${msg}`,
      );
    }
  }
  return out;
}

/** The Fix flow's user-message tail. The old text said "fix THIS error" —
 *  singular — while buildFixPayload was handing over a NUMBERED LIST; small
 *  models obeyed the verb, patched one item and stopped (the "Fix didn't fix
 *  everything" report). Plural payloads now get an explicit all-N contract
 *  with a self-check clause; big lists nudge toward a full rewrite (legal in
 *  both edit modes).
 */
export function fixInstruction(payload: string, lang: "zh" | "en", how: string): string {
  const n = (payload.match(/^\d+\. /gm) || []).length;
  if (n < 2) {
    return lang === "zh"
      ? `它在浏览器中运行时报错：${payload}\n请修复这个错误${how}。`
      : `It throws this runtime error: ${payload}\nFix this error ${how}.`;
  }
  const rewriteNudge =
    n >= 3
      ? lang === "zh"
        ? "问题较多时,直接返回完整修正后的 HTML 更稳妥。"
        : "With this many issues, returning the complete corrected HTML is the safer route."
      : "";
  return lang === "zh"
    ? `它在浏览器中运行时报出以下 ${n} 个问题：\n${payload}\n请一次性修复全部 ${n} 个问题${how}——逐项对照编号处理,输出前自查每一项都已解决,不要只修其中一项。${rewriteNudge}`
    : `It reports the following ${n} problems at runtime:\n${payload}\nFix ALL ${n} of them in this single pass ${how} — work through the numbered list item by item, and before answering verify each one is resolved; do not fix just one. ${rewriteNudge}`;
}
