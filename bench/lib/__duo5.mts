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
console.error(await call("browser_eval", { expression: `
 (() => {
   const out = [];
   for (const el of document.querySelectorAll('*')) {
     const t = (el.textContent || '').trim();
     if (!t.startsWith('START')) continue;
     const cs = getComputedStyle(el);
     out.push({ tag: el.tagName, cursor: cs.cursor, tabindex: el.getAttribute('tabindex'),
       dataTest: el.getAttribute('data-test'), role: el.getAttribute('role'),
       kids: el.children.length, len: t.length });
   }
   // 顺带看看整页有多少 cursor:pointer 的非语义元素(判断这个判据会不会爆炸)
   let ptr = 0, sem = 0;
   for (const el of document.querySelectorAll('*')) {
     const cs = getComputedStyle(el);
     if (cs.cursor !== 'pointer') continue;
     ptr++;
     if (el.matches('a,button,[role=button],[role=link],input,select,textarea,summary')) sem++;
   }
   return JSON.stringify({ start: out.slice(0, 4), pointerTotal: ptr, pointerSemantic: sem }, null, 1);
 })()` }));
await call("browser_close", {});
process.exit(0);
