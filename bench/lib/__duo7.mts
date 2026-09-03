import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (c: string, a: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(c, a); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 200); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
const url = () => call("browser_eval", { expression: "location.pathname" });
await call("browser_navigate", { url: "https://www.duolingo.com/learn" });
await wait(3000);
console.error("起始 URL:", await url());
await call("browser_click", { text: "Lesson" });
await wait(2500);
console.error("点 Lesson 后:", await url());
console.error("点 START:", (await call("browser_click", { text: "START" })).slice(0, 120));
for (const ms of [2000, 4000, 6000]) {
  await wait(ms === 2000 ? 2000 : 2000);
  console.error(`  +${ms}ms URL:`, await url());
}
const page = await call("browser_read");
const head = page.split("\n").filter((l) => l.trim()).slice(0, 12).join(" | ");
console.error("\n页面前几行:", head.slice(0, 400));
await call("browser_close", {});
process.exit(0);
