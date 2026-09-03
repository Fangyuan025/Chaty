import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (cmd: string, args: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(cmd, args); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 300); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
await call("browser_navigate", { url: "https://www.duolingo.com/learn" });
await wait(3000);
await call("browser_click", { text: "Lesson" });
await wait(2500);
console.error("=== START 元素的真实结构 ===");
console.error(await call("browser_eval", { expression: `
  (() => {
    const hits = [];
    document.querySelectorAll('*').forEach(el => {
      const t = (el.textContent || '').trim();
      if (t.startsWith('START') && t.length < 40 && el.children.length <= 2) {
        hits.push({ tag: el.tagName, role: el.getAttribute('role'), href: el.getAttribute('href'),
                    cls: (el.className||'').toString().slice(0,40), text: t });
      }
    });
    return JSON.stringify(hits.slice(0, 5), null, 1);
  })()
` }));
await call("browser_close", {});
process.exit(0);
