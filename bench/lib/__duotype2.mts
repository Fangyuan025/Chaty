import { Bridge } from "./bridge.mts";
const b = new Bridge("/Users/stevenlin/Desktop/Chaty-repo/src-tauri/target/release/chaty-headless");
const call = async (c: string, a: Record<string, unknown> = {}) => {
  try { const r: any = await b.call(c, a); return typeof r === "string" ? r : JSON.stringify(r); }
  catch (e) { return "ERR: " + String((e as Error).message).slice(0, 140); }
};
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
const btns = (p: string) => [...p.matchAll(/按钮: "([^"]+)"/g)].map((m) => m[1]);
const nextBtnState = () => call("browser_eval", { expression:
  "(()=>{const b=document.querySelector('[data-test=player-next]');if(!b)return '无';return 'aria-disabled='+(b.getAttribute('aria-disabled'))})()" });

await call("browser_navigate", { url: "https://www.duolingo.com/lesson" });
await wait(7000);
for (let i = 0; i < 15; i++) {
  const kind = await call("browser_eval", { expression:
    "(()=>{const c=document.querySelector('[data-test^=challenge]');return c?(c.getAttribute('data-test')||''):''})()" });
  const hasInput = (await call("browser_eval", { expression:
    "(!!document.querySelector('[data-test=challenge-translate-input]')).toString()" })).includes("true");
  console.error(`  轮 ${i + 1}: ${kind.replace(/"/g, "").slice(0, 40)}${hasInput ? "  ← 有输入框" : ""}`);
  if (hasInput) {
    console.error("  输入前:", await nextBtnState());
    console.error("  输入 →", (await call("browser_type", { text: "I do not know" })).slice(0, 50));
    await wait(1200);
    console.error("  框内值:", await call("browser_eval", { expression:
      "JSON.stringify((document.querySelector('[data-test=challenge-translate-input]')||{}).value)" }));
    console.error("  输入后:", await nextBtnState());
    break;
  }
  const page = await call("browser_read");
  const c = btns(page).find((t) => /^\d\s/.test(t)) || btns(page).find((t) => !/SKIP|CHECK|TOO EASY/i.test(t));
  if (c) { await call("browser_click", { text: c }); await wait(800); }
  await call("browser_click", { text: "CHECK" }); await wait(2000);
  const g = await call("browser_read");
  const cont = btns(g).find((t) => /CONTINUE|GOT IT/i.test(t));
  if (cont) { await call("browser_click", { text: cont }); await wait(2000); }
}
await call("browser_close", {});
process.exit(0);
