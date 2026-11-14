import { describe, expect, it } from "vitest";
import { closeBlocks, diagramCandidates, finishedDiagram, longestDrawableStart, repairDiagram, salvageDiagram } from "./mermaidStream";

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
      "  end",
    ]);
    expect(closeBlocks(["sequenceDiagram", "  loop every minute", "    A->>B: ping", "    alt ok"])).toHaveLength(6);
    expect(closeBlocks(["flowchart TD", "  subgraph S", "  end", "  A --> B"])).toHaveLength(4);
  });

  it("tries everything finished first, then one line fewer at a time, three at most", () => {
    const c = diagramCandidates("flowchart TD\n  subgraph S\n    A --> B\n    B --> C\n    C --> D");
    expect(c).toHaveLength(4);
    expect(c[0].endsWith("C --> D\n  end")).toBe(true);
    expect(c[3]).toBe("flowchart TD\n  subgraph S\n  end");
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
    // An id in another script keeps its bracketed title as the title.
    expect(fix("  subgraph 上升期 [Plateau Phase<br/>平台]")).toBe('  subgraph 上升期 ["Plateau Phase<br/>平台"]');
    expect(fix("  subgraph 准备阶段 [Preparation]")).toBe("  subgraph 准备阶段 [Preparation]");
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

/** Mermaid as far as these tests need it: comments and leading blank lines
 *  taken off before the lines are numbered, as Mermaid's preprocessing does;
 *  a flowchart's subgraphs must close; a line with BAD in it stops the parse
 *  there, and one with LATE stops it only on the line after (as a missing
 *  bracket does). */
async function fakeParse(text: string): Promise<void> {
  const lines = text.replace(/^\s*%%(?!\{)[^\n]+\n?/gm, "").trimStart().split("\n");
  if (!/^(graph|flowchart)\b/.test(lines[0])) throw new Error("Parse error on line 1: no diagram type");
  let depth = 0;
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (/BAD/.test(l)) throw new Error(`Parse error on line ${i + 1}:\n...`);
    if (i > 0 && /LATE/.test(lines[i - 1])) throw new Error(`Parse error on line ${i + 1}:\n...`);
    if (/^\s*subgraph\b/.test(l)) depth++;
    else if (/^\s*end\s*$/.test(l)) {
      if (depth === 0) throw new Error(`Parse error on line ${i + 1}: unexpected end`);
      depth--;
    }
  }
  if (depth) throw new Error(`Parse error on line ${lines.length}: Expecting 'end', got 'EOF'`);
}

describe("a diagram wrong in a line or two is drawn without them", () => {
  it("a last subgraph never closed is closed — nothing left out (the diagram from #mermaid report)", async () => {
    const code = [
      "graph TD",
      "    Start[开始] --> Arousal[兴奋期]",
      "    ",
      "    subgraph 准备阶段 [Preparation]",
      "        A1 --> A2",
      "    end",
      "",
      "    subgraph 消退期 [Resolution Phase<br/>恢复]",
      "        P2 --> Relax",
      "        Relax --> End[平静]",
      "        ",
      "    %% 分支说明",
      "    Note1[个体差异] -.-> Start",
      "    Type1[刺激类型] -.-> Arousal",
    ].join("\n");
    await expect(fakeParse(code)).rejects.toThrow();
    const r = await salvageDiagram(code, fakeParse);
    expect(r).not.toBeNull();
    expect(r!.skipped).toEqual([]);
    expect(r!.text).toContain("Type1[刺激类型] -.-> Arousal");
    // Closed where its indentation ends — before the lines written after it
    // at the subgraph's own depth, which the model meant to be outside it.
    const lines = r!.text.split("\n");
    const close = lines.findIndex((l, i) => i > lines.findIndex((x) => x.includes("Relax --> End")) && l.trim() === "end");
    expect(close).toBeGreaterThan(0);
    expect(close).toBeLessThan(lines.findIndex((l) => l.includes("Note1")));
    await expect(fakeParse(r!.text)).resolves.toBeUndefined();
  });

  it("a wrong line in the middle is left out and the lines after it kept", async () => {
    const code = "flowchart LR\n  A --> B\n  %% a comment shifts Mermaid's numbering\n  B --> C\n  C BAD\n  C --> D\n  D --> E";
    const r = await salvageDiagram(code, fakeParse);
    expect(r!.skipped).toEqual(["  C BAD"]);
    expect(r!.text).toContain("D --> E");
    expect(r!.text).toContain("C --> D");
  });

  it("an error noticed a line late takes out the line that caused it, not the good one after", async () => {
    const code = "flowchart LR\n  A --> B\n  B LATE\n  C --> D\n  D --> E";
    const r = await salvageDiagram(code, fakeParse);
    expect(r!.skipped).toEqual(["  B LATE"]);
    expect(r!.text).toContain("C --> D");
  });

  it("an end that closes nothing goes; a diagram whose first line is wrong is not drawn", async () => {
    const r = await salvageDiagram("flowchart TD\n  A --> B\n  end\n  B --> C", fakeParse);
    expect(r!.skipped).toEqual(["  end"]);
    expect(r!.text).toContain("B --> C");
    expect(await salvageDiagram("flowchar TD\n  A --> B", fakeParse)).toBeNull();
  });

  it("the longest start that draws is what was seen being built", async () => {
    const s = await longestDrawableStart("flowchart TD\n  A --> B\n  subgraph S\n    B --> C\n  C BAD\n  D --> E", fakeParse);
    expect(s!.lines).toBe(4);
    expect(s!.text.endsWith("B --> C\n  end")).toBe(true);
  });

  it("a flowchart node called loop at the start of a line opens no block", () => {
    expect(closeBlocks(["flowchart TD", "  loop --> A"])).toHaveLength(2);
  });

  it("an unindented block left open is closed at the end, nothing being known of where it ended", () => {
    expect(closeBlocks(["flowchart TD", "subgraph S", "A --> B", "C --> D"])).toEqual(["flowchart TD", "subgraph S", "A --> B", "C --> D", "end"]);
  });
});
