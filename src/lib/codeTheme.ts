// Switchable highlight.js palette for chat code blocks (Settings → Chat).
// The stylesheets are bundled as strings and swapped in a single <style> tag —
// static imports would all apply at once and fight over the same .hljs rules.
import githubDark from "highlight.js/styles/github-dark.css?inline";
import atomOneDark from "highlight.js/styles/atom-one-dark.css?inline";
import monokai from "highlight.js/styles/monokai.css?inline";
import nord from "highlight.js/styles/nord.css?inline";

export const CODE_THEMES = {
  "github-dark": { label: "GitHub", css: githubDark },
  "atom-one-dark": { label: "Atom One", css: atomOneDark },
  monokai: { label: "Monokai", css: monokai },
  nord: { label: "Nord", css: nord },
} as const;

export type CodeTheme = keyof typeof CODE_THEMES;

/** Swap the active highlight palette and mirror its canvas colour into
 *  `--code-bg`, which `.bubble pre` uses as the block background — so every
 *  palette keeps its intended dark canvas even under the light app themes. */
export function applyCodeTheme(name: CodeTheme): void {
  const theme = CODE_THEMES[name] ?? CODE_THEMES["github-dark"];
  let el = document.getElementById("code-theme") as HTMLStyleElement | null;
  if (!el) {
    el = document.createElement("style");
    el.id = "code-theme";
    document.head.appendChild(el);
  }
  el.textContent = theme.css;
  const bg = /\.hljs\s*\{[^}]*?background(?:-color)?:\s*([^;}]+)/.exec(theme.css)?.[1]?.trim();
  document.documentElement.style.setProperty("--code-bg", bg ?? "");
}
