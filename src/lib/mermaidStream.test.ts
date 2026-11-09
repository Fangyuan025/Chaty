import { describe, expect, it } from "vitest";
import { closeBlocks, diagramCandidates, finishedDiagram } from "./mermaidStream";

describe("a diagram being written", () => {
  it("is drawn from its finished lines — not again for each token of the line being typed", () => {
    // As it reaches the renderer: markdown's newline at the end.
    const a = finishedDiagram("flowchart TD\n  A[读代码] --> B[写\n");
    const b = finishedDiagram("flowchart TD\n  A[读代码] --> B[写测试]\n");
    expect(a).toBe("flowchart TD");
    expect(b).toBe("flowchart TD");
    expect(finishedDiagram("flowchart TD\n  A --> B\n  B --\n")).toBe("flowchart TD\n  A --> B");
  });

  it("an open subgraph or loop is closed for the drawing", () => {
    expect(closeBlocks(["flowchart TD", "  subgraph S[阶段]", "    A --> B"])).toEqual([
      "flowchart TD",
      "  subgraph S[阶段]",
      "    A --> B",
      "end",
    ]);
    expect(closeBlocks(["sequenceDiagram", "  loop every minute", "    A->>B: ping", "    alt ok"])).toHaveLength(6);
    expect(closeBlocks(["flowchart TD", "  subgraph S", "  end", "  A --> B"])).toHaveLength(4);
  });

  it("tries everything finished first, then one line fewer at a time, three at most", () => {
    const c = diagramCandidates("flowchart TD\n  subgraph S\n    A --> B\n    B --> C\n    C --> D");
    expect(c).toHaveLength(4);
    expect(c[0].endsWith("C --> D\nend")).toBe(true);
    expect(c[3]).toBe("flowchart TD\n  subgraph S\nend");
  });
});
