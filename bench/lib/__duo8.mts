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
   // 复刻 click_once 的候选选择与排序
   const sel = "a,button,[role=button],[role=link],[role=menuitem],[role=tab],input[type=submit],input[type=button],[onclick],summary,label";
   const vis = e => { const r = e.getBoundingClientRect(); if (r.width<2||r.height<2) return false;
     const s = getComputedStyle(e); return s.visibility!=='hidden'&&s.display!=='none'&&s.pointerEvents!=='none'; };
   const txts = e => [(e.innerText||e.textContent||e.value||''), e.getAttribute('aria-label')||'']
     .map(x => (x+'').trim().replace(/\\s+/g,' ').toLowerCase()).filter(Boolean);
   const cand = [...document.querySelectorAll(sel)].filter(vis);
   const hit = cand.find(e => txts(e).some(s => s.lastIndexOf('start',0)===0));
   if (!hit) return JSON.stringify({ found:false, candidates: cand.length });
   const r = hit.getBoundingClientRect();
   const x = Math.round(r.left+r.width/2), y = Math.round(r.top+r.height/2);
   const at = document.elementFromPoint(x,y);
   return JSON.stringify({
     picked: { tag:hit.tagName, role:hit.getAttribute('role'), text:(hit.innerText||'').trim().slice(0,30),
               rect:{x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width),h:Math.round(r.height)} },
     clickPoint: {x,y},
     elementAtPoint: at ? { tag:at.tagName, role:at.getAttribute('role'), text:(at.innerText||'').trim().slice(0,30) } : null,
     sameOrInside: at ? (hit===at || hit.contains(at) || at.contains(hit)) : false,
     viewport: { w: innerWidth, h: innerHeight }
   }, null, 1);
 })()` }));
await call("browser_close", {});
process.exit(0);
