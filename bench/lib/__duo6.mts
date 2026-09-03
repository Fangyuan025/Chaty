import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (c: string, a: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(c, a); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 200); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
await call("browser_navigate", { url: "https://www.duolingo.com/learn" });
await wait(3000);
await call("browser_click", { text: "Lesson" });
await wait(2500);
console.error("按文字点 START:", (await call("browser_click", { text: "START" })).slice(0, 250));
await wait(6000);
const page = await call("browser_read");
console.error("\n=== 点击后页面开头 ===\n" + page.slice(0, 600));
await call("browser_close", {});
process.exit(0);
