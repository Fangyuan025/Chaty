import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ReadmeView, readmeUrl } from "./ReadmeView";

const repo = "google/gemma-4-12b-qat-GGUF";
const render = (text: string) => renderToStaticMarkup(createElement(ReadmeView, { text, repo }));

describe("a model's README in the store", () => {
  it("renders the HTML in it — lists, tables, details, centred banners", () => {
    const html = render(
      [
        "<div><ul><li>NEW: tool calling</li><li>Vision</li></ul></div>",
        "",
        "<table><tr><th>Benchmark</th><th>Score</th></tr><tr><td>MMLU</td><td>78.3</td></tr></table>",
        "",
        "<details><summary>Sampling</summary>",
        "",
        "Use `top_p = 0.95`.",
        "",
        "</details>",
        "",
        '<p align="center"><img src="assets/banner.png" width="70%"></p>',
        "",
        "| Quant | Size |",
        "| --- | ---: |",
        "| Q4_0 | 7.15 GB |",
      ].join("\n"),
    );
    expect(html.match(/<li>/g)).toHaveLength(2);
    expect(html.match(/<table>/g)).toHaveLength(2);
    expect(html).toContain("<details><summary>Sampling</summary>");
    expect(html).toContain('<p align="center">');
    expect(html).toContain(`src="https://huggingface.co/${repo}/resolve/main/assets/banner.png"`);
    expect(html).toContain('width="70%"');
  });

  it("takes out what could run, restyle the app or leave CSS behind", () => {
    const html = render(
      [
        "<script>window.__pwned = 1</script>",
        "<style>body { display: none }</style>",
        '<img src="javascript:alert(1)" onerror="alert(2)">',
        '<a href="javascript:alert(3)" onclick="alert(4)">click</a>',
        '<p style="position: fixed; inset: 0">cover</p>',
        '<iframe src="https://example.com"></iframe>',
        '<svg><text>svg text</text></svg>',
      ].join("\n"),
    );
    expect(html).not.toMatch(/<script|<style|<iframe|<svg/i);
    expect(html).not.toMatch(/pwned|display: none|svg text/);
    expect(html).not.toMatch(/javascript:|onerror|onclick|style=/i);
    expect(html).not.toContain("<img");
    expect(html).toContain("click");
    expect(html).toContain("cover");
  });

  it("draws GitHub's alerts as callouts, and leaves other quotes alone", () => {
    const html = render(
      [
        "> [!Note]",
        "> This repository contains model weights.",
        ">",
        "> These artifacts are compatible.",
        "",
        "> [!WARNING]",
        ">",
        "> Needs 24 GB.",
        "",
        "> Just a quote, [!TIP] inside it.",
      ].join("\n"),
    );
    expect(html).toContain('<blockquote class="readme-alert readme-alert-note">');
    expect(html).toContain('<p class="readme-alert-title">Note</p>');
    expect(html).toContain("<p>This repository contains model weights.</p>");
    expect(html).toContain('<blockquote class="readme-alert readme-alert-warning">');
    expect(html).toContain('<p class="readme-alert-title">Warning</p>');
    expect(html).not.toContain("<p></p>");
    expect(html).not.toMatch(/\[!(Note|WARNING)\]/);
    expect(html).toContain("<blockquote>\n<p>Just a quote, [!TIP] inside it.</p>");
    // A class written by the README author is still taken out.
    expect(render('<blockquote class="readme-alert-caution">x</blockquote>')).not.toContain("class=");
  });

  it("points relative links into the repo and keeps absolute ones", () => {
    expect(readmeUrl("LICENSE", "href", repo)).toBe(`https://huggingface.co/${repo}/blob/main/LICENSE`);
    expect(readmeUrl("./assets/chart.png", "src", repo)).toBe(`https://huggingface.co/${repo}/resolve/main/assets/chart.png`);
    expect(readmeUrl("/docs/guide.md", "href", repo)).toBe(`https://huggingface.co/${repo}/blob/main/docs/guide.md`);
    expect(readmeUrl("https://example.com/a?b=c", "href", repo)).toBe("https://example.com/a?b=c");
    expect(readmeUrl("//example.com/x.png", "src", repo)).toBe("https://example.com/x.png");
    expect(readmeUrl("#usage", "href", repo)).toBe("#usage");
    expect(readmeUrl("javascript:alert(1)", "href", repo)).toBe("");
    expect(readmeUrl("data:image/png;base64,AAAA", "src", repo)).toBe("");
  });
});
