import { describe, expect, test } from "vitest";
import { buildScanView, parseStreamPatches } from "./canvasStream";

const BASE = ["<html>", "<body>", "<h1>Old title</h1>", "<p>keep me</p>", "<button>Go</button>", "</body>", "</html>"].join("\n");

describe("parseStreamPatches", () => {
  test("completed and in-flight blocks are separated", () => {
    const acc = [
      "<<<<<<< SEARCH",
      "<h1>Old title</h1>",
      "=======",
      "<h1>New title</h1>",
      ">>>>>>> REPLACE",
      "<<<<<<< SEARCH",
      "<button>Go</button>",
      "=======",
      "<button>Sta",
    ].join("\n");
    const p = parseStreamPatches(acc);
    expect(p.done).toHaveLength(1);
    expect(p.done[0].replace).toBe("<h1>New title</h1>");
    expect(p.active?.search).toBe("<button>Go</button>");
    expect(p.active?.replace).toBe("<button>Sta");
  });

  test("SEARCH still being copied has replace=null", () => {
    const acc = "<<<<<<< SEARCH\n<h1>Old ti";
    const p = parseStreamPatches(acc);
    expect(p.done).toHaveLength(0);
    expect(p.active?.replace).toBeNull();
  });
});

describe("buildScanView — patch mode", () => {
  test("resolved patch shows del+add rows with scan on the newest change", () => {
    const acc = "<<<<<<< SEARCH\n<h1>Old title</h1>\n=======\n<h1>New title</h1>\n>>>>>>> REPLACE";
    const v = buildScanView(BASE, acc);
    expect(v.mode).toBe("patch");
    const kinds = v.rows.map((r) => r.kind).join(",");
    expect(kinds).toContain("del,add");
    expect(v.rows[v.scanIndex!].text).toContain("New title");
  });

  test("mid-REPLACE stream applies the partial replacement live", () => {
    const acc = "<<<<<<< SEARCH\n<h1>Old title</h1>\n=======\n<h1>Half";
    const v = buildScanView(BASE, acc);
    expect(v.rows.some((r) => r.kind === "add" && r.text.includes("Half"))).toBe(true);
  });

  test("mid-SEARCH stream anchors the scan on the matching source line", () => {
    const acc = "<<<<<<< SEARCH\n<button>Go</button>";
    const v = buildScanView(BASE, acc);
    expect(v.scanIndex).not.toBeNull();
    expect(v.rows[v.scanIndex!].text).toContain("<button>Go</button>");
  });

  test("prose before any block = waiting on the untouched code", () => {
    const v = buildScanView(BASE, "Let me think about this…");
    expect(v.mode).toBe("waiting");
    expect(v.rows.every((r) => r.kind === "ctx")).toBe(true);
  });

  // The follow-scroll UX depends on this invariant: once any block has
  // started, the head must exist at EVERY stream offset (between-block gaps
  // fall back to the last changed row). Sweeps the whole stream so a parser
  // tweak can't quietly strand the viewport mid-generation.
  test("scan head exists at every offset once patching starts", () => {
    const stream = [
      "Warm the copy first, then the palette.\n",
      "<<<<<<< SEARCH\n<h1>Old title</h1>\n",
      "=======\n<h1>Bright new title</h1>\n>>>>>>> REPLACE\n",
      "<<<<<<< SEARCH\n<p>keep me</p>\n",
      "=======\n<p>keep me, warmly</p>\n>>>>>>> REPLACE\n",
    ].join("");
    const firstBlock = stream.indexOf("<<<");
    for (let cut = firstBlock + 18; cut <= stream.length; cut += 7) {
      const v = buildScanView(BASE, stream.slice(0, cut));
      expect(v.mode).toBe("patch");
      expect(v.scanIndex, `offset ${cut}`).not.toBeNull();
    }
  });

  test("done-patch cache never changes results across growing streams", () => {
    const acc1 = "<<<<<<< SEARCH\n<h1>Old title</h1>\n=======\n<h1>A</h1>\n>>>>>>> REPLACE\n";
    const withActive = acc1 + "<<<<<<< SEARCH\n<button>Go</button>\n=======\n<button>Stop";
    const a = buildScanView(BASE, withActive);
    // Same call again (cache warm) and after an unrelated base (cache bust).
    buildScanView("<html>\n<p>other</p>\n</html>", "x");
    const b = buildScanView(BASE, withActive);
    expect(b.rows).toEqual(a.rows);
    expect(b.scanIndex).toBe(a.scanIndex);
  });
});

describe("buildScanView — full-document mode", () => {
  test("unstreamed old tail renders as pending, not deleted", () => {
    const acc = "```html\n<html>\n<body>\n<h1>Brand new</h1>";
    const v = buildScanView(BASE, acc);
    expect(v.mode).toBe("full");
    expect(v.rows.some((r) => r.kind === "pending")).toBe(true);
    const last = v.rows[v.rows.length - 1];
    expect(last.kind).toBe("pending");
    expect(last.text).toBe("</html>");
    expect(v.rows.some((r) => r.kind === "add" && r.text.includes("Brand new"))).toBe(true);
  });
});
