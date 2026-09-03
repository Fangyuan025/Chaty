import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (c: string, a: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(c, a); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 150); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
const els = (p: string) => [...p.matchAll(/按钮: "([^"]+)"/g)].map((m) => m[1]);

await call("browser_navigate", { url: "https://www.duolingo.com/lesson" });
await wait(7000);
for (let q = 1; q <= 3; q++) {
  const page = await call("browser_read");
  const body = page.slice(0, page.indexOf("可交互元素"));
  const buttons = els(page);
  const choices = buttons.filter((t) => /^\d\s/.test(t));
  console.error(`\n── 第 ${q} 题  题干: ${body.split("\n").filter(Boolean).slice(1, 6).join(" ").slice(0, 90)}`);
  console.error(`   可点: ${JSON.stringify(buttons.slice(0, 6))}`);
  if (!choices.length) { console.error("   (非选择题,停在这里)"); break; }
  console.error("   选:", choices[choices.length - 1], "→", (await call("browser_click", { text: choices[choices.length - 1] })).slice(0, 60));
  await wait(1200);
  const after = await call("browser_read");
  console.error("   选中后 CHECK 是否可用:", /按钮: "CHECK"/.test(after) ? "在列表里" : "不在");
  console.error("   CHECK →", (await call("browser_click", { text: "CHECK" })).slice(0, 60));
  await wait(2500);
  const graded = await call("browser_read");
  const verdict = /Correct|正确|Nice|Excellent/i.test(graded) ? "判对" : /Correct solution|Incorrect|错误/i.test(graded) ? "判错" : "?";
  console.error("   判定:", verdict, "| 下一步按钮:", JSON.stringify(els(graded).slice(0, 4)));
  const cont = els(graded).find((t) => /CONTINUE|继续|GOT IT/i.test(t));
  if (cont) { console.error("   继续 →", (await call("browser_click", { text: cont })).slice(0, 40)); await wait(2500); }
}
await call("browser_close", {});
process.exit(0);
