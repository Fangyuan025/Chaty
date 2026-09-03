import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (cmd: string, args: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(cmd, args); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 200); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
await call("browser_navigate", { url: "https://www.duolingo.com/learn" });
await wait(3000);

// 单元节点:多邻国点节点先出浮层,浮层里才有 START
console.error("点第一个 Lesson:", (await call("browser_click", { text: "Lesson" })).slice(0, 200));
await wait(2500);
let page = await call("browser_read");
const idx = page.indexOf("可交互元素");
console.error("\n=== 可交互元素列表(浮层打开后) ===\n" + page.slice(idx, idx + 1200));
const starts = [...page.slice(idx).matchAll(/"([^"]*(?:START|\+\d+ XP)[^"]*)"/gi)].map((m) => m[1]);
console.error("\n匹配到的 START 类:", JSON.stringify(starts));
if (starts.length) {
  console.error("点 START:", (await call("browser_click", { text: starts[0] })).slice(0, 200));
  await wait(6000);
  page = await call("browser_read");
}
console.error("\n=== 当前页面开头 800 字 ===\n" + page.slice(0, 800));
await call("browser_close", {});
process.exit(0);
