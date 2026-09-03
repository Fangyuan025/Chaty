/** Drive the agent's own browser tools against the real, signed-in site. */
import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const step = async (name: string, cmd: string, args: Record<string, unknown> = {}) => {
  const t0 = Date.now();
  try {
    const r: any = await b.call(cmd, args);
    const s = typeof r === "string" ? r : JSON.stringify(r);
    console.error(`\n[${name}] ${Date.now() - t0}ms\n${s.slice(0, 1400)}`);
    return s;
  } catch (e) {
    console.error(`\n[${name}] 失败: ${String((e as Error).message).slice(0, 300)}`);
    return "";
  }
};
await step("导航到多邻国", "browser_navigate", { url: "https://www.duolingo.com/learn" });
await step("读取页面", "browser_read", {});
process.exit(0);
