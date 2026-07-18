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
