import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (c: string, a: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(c, a); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 200); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
await call("browser_navigate", { url: "https://www.duolingo.com/lesson" });
await wait(7000);
console.error("URL:", await call("browser_eval", { expression: "location.pathname" }));
const page = await call("browser_read");
const idx = page.indexOf("可交互元素");
console.error("\n=== 题面 ===\n" + page.slice(0, idx > 0 ? Math.min(idx, 900) : 900));
console.error("\n=== 可交互元素 ===\n" + (idx > 0 ? page.slice(idx, idx + 900) : "(无)"));
await call("browser_close", {});
process.exit(0);
