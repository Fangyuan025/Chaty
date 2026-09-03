import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (c: string, a: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(c, a); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 200); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
await call("browser_navigate", { url: "https://www.duolingo.com/lesson" });
await wait(7000);
console.error(await call("browser_eval", { expression: `
 (() => {
   const rep = el => { const cs = getComputedStyle(el); const r = el.getBoundingClientRect();
     return { tag: el.tagName, role: el.getAttribute('role'), tabindex: el.getAttribute('tabindex'),
       cursor: cs.cursor, kids: el.children.length,
       text: (el.innerText||'').trim().replace(/\\s+/g,' ').slice(0, 24),
       size: Math.round(r.width) + 'x' + Math.round(r.height),
       semantic: el.matches('a,button,[role=button],[role=link],[role=menuitem],[role=tab],input,textarea,select,summary') };
   };
   const pick = sel => [...document.querySelectorAll(sel)].slice(0, 3).map(rep);
   return JSON.stringify({
     choices: pick('[data-test="challenge-choice"]'),
     next: pick('[data-test="player-next"]'),
     hints: pick('[data-test="hint-token"]'),
   }, null, 1);
 })()` }));
await call("browser_close", {});
process.exit(0);
