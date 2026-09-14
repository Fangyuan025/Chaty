/** Where the tokens of an edit-heavy run went. Reads the per-task transcripts
 *  the coder runner writes with CHATY_BENCH_TRANSCRIPT=<dir> and classifies
 *  every model round:
 *    ok        a tool call that parsed and ran
 *    edit-miss an edit that ran but failed (old_string not found / not unique)
 *    xml       the model wrote its native <function=…><parameter=…> call
 *    bad-json  a <tool_call> that could not be parsed
 *    empty     a call with no arguments
 *    answer    no tool call (a final answer or prose)
 *  and totals the characters each class cost.
 *    npx tsx bench/coder/__editstats.mts <transcript-dir> [...more dirs]
 */
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";

type Ev = { ev: string; t?: string; call?: { name: string; args: Record<string, unknown> }; status?: string; result?: string };

const EDIT_TOOLS = new Set(["edit_file", "multi_edit", "write_file", "edit_lines"]);
const MISS = /not found|未找到|not unique|不唯一|anchor|锚点不匹配|no such|找不到/i;
const INVALID = /格式无效|not valid|无法解析|完全相同的无效调用|same invalid call/i;
// The loop's own corrections for a call with an argument missing: the
// ladder's first note, its second ("…in a row without"), and the disable.
const EMPTY = /缺少 "|missing "|连续两次发出没有|in a row without|空参数|empty-args|缺少必需参数|missing required/i;

for (const dir of process.argv.slice(2)) {
  for (const f of readdirSync(dir).filter((n) => n.endsWith(".jsonl"))) {
    const evs: Ev[] = readFileSync(path.join(dir, f), "utf8")
      .split("\n")
      .filter(Boolean)
      .map((l) => JSON.parse(l));
    const cost: Record<string, { n: number; chars: number }> = {};
    const add = (k: string, chars: number) => {
      cost[k] ??= { n: 0, chars: 0 };
      cost[k].n++;
      cost[k].chars += chars;
    };
    // A raw round is judged by what follows it: a step card (ran), an
    // invalid-call correction, or nothing (an answer).
    for (let i = 0; i < evs.length; i++) {
      const e = evs[i];
      if (e.ev !== "raw") continue;
      const raw = e.t ?? "";
      const next = evs.slice(i + 1).find((x) => x.ev === "raw" || x.ev === "step" || x.ev === "inject" || x.ev === "final");
      let k = "answer";
      if (next?.ev === "step") {
        k = next.status === "error" && EDIT_TOOLS.has(next.call?.name ?? "") && MISS.test(next.result ?? "") ? "edit-miss" : "ok";
        if (next.status === "error" && EMPTY.test(next.result ?? "")) k = "empty";
      } else if (next?.ev === "inject" && INVALID.test(next.t ?? "")) {
        k = /<function=/.test(raw) ? "xml" : "bad-json";
      } else if (next?.ev === "inject" && EMPTY.test(next.t ?? "")) {
        k = "empty";
      } else if (/<function=update_plan>|call:update_plan|"name"\s*:\s*"update_plan"/.test(raw)) {
        // update_plan runs in the loop itself and leaves no step card — a
        // call that ran, not a wasted round.
        k = "ok";
      } else if (/<tool_call>|<function=/.test(raw)) {
        k = /<function=/.test(raw) ? "xml" : "bad-json";
      }
      add(k, raw.length);
    }
    const final = evs.filter((e) => e.ev === "final").pop();
    const total = Object.values(cost).reduce((a, c) => a + c.chars, 0);
    const wasted = ["edit-miss", "xml", "bad-json", "empty"].reduce((a, k) => a + (cost[k]?.chars ?? 0), 0);
    console.log(
      `${path.basename(dir)}/${f.replace(".jsonl", "")}: ` +
        Object.entries(cost)
          .map(([k, c]) => `${k}=${c.n}(${c.chars}c)`)
          .join(" ") +
        ` | wasted ${wasted}/${total} chars (${total ? Math.round((wasted / total) * 100) : 0}%) | end=${final?.ev === "final" ? "final" : "-"}`,
    );
  }
}
