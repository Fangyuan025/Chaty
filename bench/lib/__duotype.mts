import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (c: string, a: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(c, a); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 150); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
await call("browser_navigate", { url: "https://www.duolingo.com/lesson" });
await wait(7000);
// 走到有输入框的题
for (let i = 0; i < 12; i++) {
  const has = await call("browser_eval", { expression:
    "(!!document.querySelector('textarea,input[type=text]')).toString()" });
  if (has.includes("true")) break;
  const page = await call("browser_read");
  const c = [...page.matchAll(/按钮: "([^"]+)"/g)].map((m) => m[1]).find((t) => /^\d\s/.test(t));
  if (c) { await call("browser_click", { text: c }); await wait(900); }
  await call("browser_click", { text: "CHECK" }); await wait(2200);
  const g = await call("browser_read");
  const cont = [...g.matchAll(/按钮: "([^"]+)"/g)].map((m) => m[1]).find((t) => /CONTINUE|GOT IT/i.test(t));
  if (cont) { await call("browser_click", { text: cont }); await wait(2200); }
}
const page = await call("browser_read");
console.error("题面:", page.slice(0, page.indexOf("可交互元素")).split("\n").filter(Boolean).slice(1, 7).join(" ").slice(0, 140));
console.error("\n输入前 CHECK 状态:", await call("browser_eval", { expression:
  "(()=>{const b=document.querySelector('[data-test=player-next]');return b?(b.getAttribute('aria-disabled')||b.disabled||'enabled')+'':'无'})()" }));
console.error("输入 →", (await call("browser_type", { text: "I do not know my mother-in-law" })).slice(0, 80));
await wait(1200);
console.error("输入后框内的值:", await call("browser_eval", { expression:
  "(()=>{const f=document.querySelector('textarea,input[type=text]');return f?JSON.stringify(f.value):'无'})()" }));
console.error("输入后 CHECK 状态:", await call("browser_eval", { expression:
  "(()=>{const b=document.querySelector('[data-test=player-next]');return b?(b.getAttribute('aria-disabled')||b.disabled||'enabled')+'':'无'})()" }));
await call("browser_close", {});
process.exit(0);
