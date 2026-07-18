// Switchable highlight.js palette for chat code blocks (Settings → Chat).
// The stylesheets are bundled as strings and swapped in a single <style> tag —
// static imports would all apply at once and fight over the same .hljs rules.
import githubDark from "highlight.js/styles/github-dark.css?inline";
import githubLight from "highlight.js/styles/github.css?inline";
import atomOneDark from "highlight.js/styles/atom-one-dark.css?inline";
import atomOneLight from "highlight.js/styles/atom-one-light.css?inline";
import monokai from "highlight.js/styles/monokai.css?inline";
import nord from "highlight.js/styles/nord.css?inline";

/** `light` is the palette's sibling for light app themes. Monokai and Nord
 *  have no official light variant — they keep their dark canvas everywhere
 *  (the block background follows the palette via --code-bg either way). */
type ThemeDef = { label: string; css: string; light?: string };
export const CODE_THEMES: Record<"github-dark" | "atom-one-dark" | "monokai" | "nord", ThemeDef> = {
  "github-dark": { label: "GitHub", css: githubDark, light: githubLight },
  "atom-one-dark": { label: "Atom One", css: atomOneDark, light: atomOneLight },
  monokai: { label: "Monokai", css: monokai },
  nord: { label: "Nord", css: nord },
};

export type CodeTheme = keyof typeof CODE_THEMES;

/** Swap the active highlight palette and mirror its canvas colour into
 *  `--code-bg`, which `.bubble pre` uses as the block background. Under light
 *  app themes, palettes with a light sibling switch to it — the rest keep
 *  their intended dark canvas. */
export function applyCodeTheme(name: CodeTheme, preferLight = false): void {
  const theme = CODE_THEMES[name] ?? CODE_THEMES["github-dark"];
  const css = preferLight && theme.light ? theme.light : theme.css;
  let el = document.getElementById("code-theme") as HTMLStyleElement | null;
  if (!el) {
    el = document.createElement("style");
    el.id = "code-theme";
    document.head.appendChild(el);
  }
  el.textContent = css;
  const bg = /\.hljs\s*\{[^}]*?background(?:-color)?:\s*([^;}]+)/.exec(css)?.[1]?.trim();
  document.documentElement.style.setProperty("--code-bg", bg ?? "");
}
