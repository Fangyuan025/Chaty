import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (cmd: string, args: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(cmd, args); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 200); }
};
await call("browser_navigate", { url: "https://www.duolingo.com/learn" });
console.error("=== 点 Lesson ===");
console.error((await call("browser_click", { text: "Lesson" })).slice(0, 300));
await new Promise((r) => setTimeout(r, 4000));
console.error("\n=== 练习页读到的内容 ===");
console.error((await call("browser_read")).slice(0, 2000));
console.error("\n=== 截图尺寸 ===");
console.error((await call("browser_screenshot", {})).slice(0, 200));
await call("browser_close", {});
process.exit(0);
