/** A real coding session: create, checkpoint, edit several ways, then undo.
 *  Undo is the one operation where a defect costs the user their work. */
import { Bridge } from "./bridge.mts";
import { readFileSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
const ws = process.argv[2];
rmSync(ws, { recursive: true, force: true }); mkdirSync(ws, { recursive: true });
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
await b.call("agent_set_workspace", { path: ws });

const check = (n: string, ok: boolean, d = "") => console.error(`${ok ? "✓" : "✗ 缺陷"} ${n}${d ? "\n    " + d : ""}`);
const read = (f: string) => (existsSync(`${ws}/${f}`) ? readFileSync(`${ws}/${f}`, "utf8") : null);

// 会话开始前就存在的文件(含中文,字节偏移的雷区)
writeFileSync(`${ws}/notes.md`, "# 笔记\n\n第一行内容\n第二行内容\n");
writeFileSync(`${ws}/app.ts`, "export function add(a: number, b: number) {\n  return a + b;\n}\n");
const before = { notes: read("notes.md")!, app: read("app.ts")! };

const cp: any = await b.call("agent_checkpoint_begin", {});
const cpId = typeof cp === "number" ? cp : cp?.id ?? cp;

// 用户在这一回合里做的事:改两个已有文件、新建一个
await b.call("agent_edit_file", { path: "notes.md", oldString: "第一行内容", newString: "第一行已改" });
await b.call("agent_multi_edit", { path: "app.ts", edits: [{ old_string: "a + b", new_string: "a - b" }] });
await b.call("agent_write_file", { path: "brand new.ts", content: "export const x = 1;\n" });

check("编辑确实生效", read("notes.md")!.includes("第一行已改") && read("app.ts")!.includes("a - b"));
check("新文件确实建了", read("brand new.ts") !== null);

// 用户点了「撤销这一回合」
const rv = await b.call("agent_checkpoint_revert_to", { id: cpId });
console.error(`    撤销返回: ${JSON.stringify(rv).slice(0, 90)}`);

check("已有文件逐字节还原(含中文)", read("notes.md") === before.notes,
  read("notes.md") === before.notes ? "" : `现在: ${JSON.stringify(read("notes.md"))}`);
check("第二个已有文件也还原", read("app.ts") === before.app,
  read("app.ts") === before.app ? "" : `现在: ${JSON.stringify(read("app.ts"))}`);
check("回合内新建的文件被删除", read("brand new.ts") === null,
  read("brand new.ts") === null ? "" : "撤销后仍存在——用户会看到一个本该消失的文件");

// 再撤销一次(用户手抖点两下)
try {
  await b.call("agent_checkpoint_revert_to", { id: cpId });
  check("重复撤销不炸", true);
} catch (e) { check("重复撤销不炸", false, String((e as Error).message).slice(0, 100)); }
check("重复撤销后文件仍正确", read("notes.md") === before.notes && read("app.ts") === before.app);
process.exit(0);
