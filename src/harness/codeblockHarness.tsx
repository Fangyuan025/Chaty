// Visual test harness for chat code-block rendering (line-number gutter,
// light/dark code palettes, scroll behavior). Real Markdown component, real
// App.css — served by plain vite at /codeblock-harness.html, no Tauri needed.
import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import "../App.css";
import { LangProvider } from "../lib/i18n";
import { Markdown } from "../components/Markdown";
import { applyCodeTheme, CODE_THEMES, type CodeTheme } from "../lib/codeTheme";

const QUICKSORT = `def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    def partition(arr, low, high):
        pivot = arr[high]  # 选最后一个元素为基准
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    if low < high:
        p = partition(arr, low, high)
        quick_sort(arr, low, p - 1)
        quick_sort(arr, p + 1, high)

    return arr`;

const FIFTY = Array.from({ length: 50 }, (_, i) => `console.log("line ${i + 1} of a very tall block");`).join("\n");

const WIDE = `const url = "https://example.com/an/extremely/long/path/that/forces/horizontal/scrolling/in/the/code/block?with=query&params=and#fragments-galore-to-make-it-even-longer-and-longer";`;

const DOC = [
  "## 版本二:原地版(空间优化,推荐)",
  "",
  "```python",
  QUICKSORT,
  "```",
  "",
  "行内代码 `arr[high]` 与 **表格**:",
  "",
  "| 方案 | 空间 | 稳定 |",
  "| --- | --- | --- |",
  "| 原地版 | O(log n) | 否 |",
  "| 简洁版 | O(n) | 否 |",
  "",
  "短块(应无行号):",
  "",
  "```js",
  "const a = 1;",
  "const b = 2;",
  "```",
  "",
  "超宽单行(横向滚动,行号应吸附不动):",
  "",
  "```js",
  WIDE,
  Array.from({ length: 6 }, (_, i) => `const pad${i} = ${i};`).join("\n"),
  "```",
  "",
  "50 行超高块(60vh 封顶,应只有一条纵向滚动条):",
  "",
  "```js",
  FIFTY,
  "```",
].join("\n");

function Harness() {
  const [light, setLight] = useState(false);
  const [palette, setPalette] = useState<CodeTheme>("github-dark");

  const apply = (l: boolean, p: CodeTheme) => {
    document.documentElement.dataset.theme = l ? "light" : "dark";
    applyCodeTheme(p, l);
    setLight(l);
    setPalette(p);
  };

  return (
    <div id="scroller" style={{ background: "var(--bg)", color: "var(--text)", height: "100vh", overflow: "auto", padding: 24, boxSizing: "border-box" }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }} id="controls">
        <button id="toggle-appearance" onClick={() => apply(!light, palette)}>
          {light ? "→ dark" : "→ light"}
        </button>
        {(Object.keys(CODE_THEMES) as CodeTheme[]).map((k) => (
          <button key={k} id={`palette-${k}`} onClick={() => apply(light, k)} style={{ fontWeight: palette === k ? 700 : 400 }}>
            {CODE_THEMES[k].label}
          </button>
        ))}
      </div>
      <div className="msg assistant" style={{ maxWidth: 760 }}>
        <div className="bubble">
          <div className="answer">
            <Markdown>{DOC}</Markdown>
          </div>
        </div>
      </div>
    </div>
  );
}

document.documentElement.dataset.theme = "dark";
applyCodeTheme("github-dark", false);
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LangProvider>
      <Harness />
    </LangProvider>
  </React.StrictMode>,
);
