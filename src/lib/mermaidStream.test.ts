import { describe, expect, it } from "vitest";
import { closeBlocks, diagramCandidates, finishedDiagram, repairDiagram } from "./mermaidStream";

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

describe("a flowchart as models write it, made to parse", () => {
  const fix = (body: string) => repairDiagram(`flowchart TD\n${body}`).split("\n").slice(1).join("\n");

  it("quotes a label with brackets, quotes or Mermaid's punctuation in it — the text kept", () => {
    expect(fix("  A[调用 foo(bar)] --> B{价格 > 5?}")).toBe('  A["调用 foo(bar)"] --> B{"价格 > 5?"}');
    expect(fix('  C[say "hi"] --> D(结果: 成功)')).toBe('  C["say #quot;hi#quot;"] --> D("结果: 成功")');
    expect(fix("  E((开始)) --> F([读取 config.json])")).toBe("  E((开始)) --> F([读取 config.json])");
    expect(fix("  G[C# 代码] --> H[第一行\\n第二行]")).toBe('  G["C#35; 代码"] --> H["第一行<br/>第二行"]');
  });

  it("a label quoted another language's way, and link text", () => {
    expect(fix('  A["他说 \\"你好\\""] --> B')).toBe('  A["他说 #quot;你好#quot;"] --> B');
    expect(fix("  A -->|是(通过)| B")).toBe('  A -->|"是(通过)"| B');
    expect(fix("  A -->|否| B")).toBe("  A -->|否| B");
  });

  it("a subgraph's spaced title, and arrows typed as characters", () => {
    expect(fix("  subgraph 输入 阶段\n    A --> B\n  end")).toBe('  subgraph "输入 阶段"\n    A --> B\n  end');
    expect(fix("  subgraph S1[阶段 (一)]")).toBe('  subgraph S1["阶段 (一)"]');
    expect(fix("  A → B\n  B —> C")).toBe("  A --> B\n  B --> C");
  });

  it("a node named end, which is Mermaid's word for closing a subgraph", () => {
    expect(fix("  start[开始] --> end[结束]\n  end --> A[the end]")).toBe("  start[开始] --> End[结束]\n  End --> A[the end]");
    expect(fix("  subgraph S\n    A --> B\n  end")).toBe("  subgraph S\n    A --> B\n  end");
  });

  it("leaves what was right alone, and other diagrams' text", () => {
    const good = 'flowchart LR\n  A["已加引号 (ok)"] --> B[普通文字]\n  style A fill:#f9f\n  classDef red fill:#f00';
    expect(repairDiagram(good)).toBe(good);
    const seq = "sequenceDiagram\n  A->>B: 调用 foo(bar)";
    expect(repairDiagram(seq)).toBe(seq);
  });
});
