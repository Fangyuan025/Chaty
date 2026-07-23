/**
 * ChatyWeb-Bench graders — one function per task id.
 *
 * A grader sees the fixture server's full state (after the agent worked) plus
 * the agent's final message, and returns pass/fail with a reason. State tasks
 * assert on server state (the single source of truth — every UI mutation goes
 * through the API); answer tasks match required substrings, case-insensitive,
 * in the final message.
 */
import type { seedState } from "./server.mts";

type State = ReturnType<typeof seedState>;
export type Verdict = { pass: boolean; why: string };
type Grader = (state: State, finalText: string) => Verdict;

const ok: Verdict = { pass: true, why: "ok" };
const fail = (why: string): Verdict => ({ pass: false, why });

/** Lowercase and strip markdown emphasis/backticks — models bold the key
 *  number ("maximum of **5** days") and plain substring matching slides off. */
function norm(s: string): string {
  return s.toLowerCase().replace(/[*_`]/g, "");
}
/** All `needles` present (case-insensitive) in the final message. */
function answerHas(finalText: string, needles: string[]): Verdict {
  const t = norm(finalText);
  const missing = needles.filter((n) => !t.includes(n.toLowerCase()));
  return missing.length ? fail(`final answer missing: ${missing.join(", ")}`) : ok;
}
/** At least one alternative present. */
function answerAny(finalText: string, alts: string[]): Verdict {
  const t = norm(finalText);
  return alts.some((n) => t.includes(n.toLowerCase()))
    ? ok
    : fail(`final answer has none of: ${alts.join(" | ")}`);
}

export const GRADERS: Record<string, Grader> = {
  // ---------- shop ----------
  "shop-cheapest-audio": (_s, t) => answerHas(t, ["onyx mic arm"]),

  "shop-add-qty2": (s) => {
    const c = s.shop.cart;
    if (c.length !== 1) return fail(`cart has ${c.length} lines, want 1`);
    if (c[0].id !== "p04" || c[0].qty !== 2) return fail(`cart is ${JSON.stringify(c)}, want p04 × 2`);
    return ok;
  },

  "shop-order-cheapest-keyboard": (s) => {
    if (s.shop.orders.length !== 1) return fail(`${s.shop.orders.length} orders, want 1`);
    const o = s.shop.orders[0] as any;
    const items = o.items as { id: string; qty: number }[];
    if (items.length !== 1 || items[0].id !== "p06" || items[0].qty !== 1)
      return fail(`order items ${JSON.stringify(items)}, want [p06 × 1]`);
    if (!String(o.name).toLowerCase().includes("riley yu")) return fail(`name "${o.name}"`);
    if (!String(o.address).toLowerCase().includes("88 pine")) return fail(`address "${o.address}"`);
    if (o.shipping !== "standard") return fail(`shipping "${o.shipping}"`);
    return ok;
  },

  "shop-price-sum": (_s, t) => answerHas(t, ["77.49"]),

  "shop-cart-cheap-cables": (s) => {
    const want = new Set(["p03", "p04", "p15", "p21"]);
    const got = new Set(s.shop.cart.map((c) => c.id));
    if (got.size !== want.size || [...want].some((id) => !got.has(id)))
      return fail(`cart ids ${[...got].join(",") || "(empty)"}, want ${[...want].join(",")}`);
    const extraQty = s.shop.cart.find((c) => c.qty !== 1);
    if (extraQty) return fail(`${extraQty.id} qty ${extraQty.qty}, want 1 each`);
    return ok;
  },

  // ---------- inbox ----------
  "inbox-discount-code": (_s, t) => answerHas(t, ["skyfall25"]),

  "inbox-reply-po": (s) => {
    const inv = s.inbox.emails.find((e) => e.id === 1)!;
    if (!inv.replies.length) return fail("no reply on the Acme invoice email");
    if (!inv.replies.some((r) => r.toUpperCase().includes("PO-7741")))
      return fail(`reply lacks PO-7741: ${JSON.stringify(inv.replies)}`);
    return ok;
  },

  "inbox-archive-newsletters": (s) => {
    const newsletters = [2, 6, 9];
    for (const id of newsletters)
      if (!s.inbox.emails.find((e) => e.id === id)!.archived) return fail(`email ${id} not archived`);
    const collateral = s.inbox.emails.filter((e) => e.archived && !newsletters.includes(e.id));
    if (collateral.length) return fail(`archived non-newsletters: ${collateral.map((e) => e.id).join(",")}`);
    return ok;
  },

  "inbox-compose-pilot": (s) => {
    const m = (s.inbox.sent as any[]).find((x) => String(x.to).includes("rene@brightpath.example"));
    if (!m) return fail("no sent mail to rene@brightpath.example");
    if (!String(m.subject).toLowerCase().includes("pilot scope")) return fail(`subject "${m.subject}"`);
    if (!String(m.body).toLowerCase().includes("q3")) return fail("body does not mention Q3");
    return ok;
  },

  // ---------- board ----------
  "board-move-loginbug": (s) => {
    const inDone = s.board.cols.done.some((c) => c.id === 2);
    const elsewhere = [...s.board.cols.todo, ...s.board.cols.doing].some((c) => c.id === 2);
    if (!inDone || elsewhere) return fail(`card 2 in done=${inDone}, still elsewhere=${elsewhere}`);
    return ok;
  },

  "board-add-card": (s) => {
    const hit = s.board.cols.doing.find(
      (c) => c.title.trim().toLowerCase() === "draft press release" && c.assignee === "Dana",
    );
    return hit ? ok : fail(`Doing column: ${JSON.stringify(s.board.cols.doing)}`);
  },

  "board-most-cards": (_s, t) => answerHas(t, ["sam"]),

  "board-clear-done": (s) => {
    if (s.board.cols.done.length) return fail(`done still has ${s.board.cols.done.length} cards`);
    const survivors = [...s.board.cols.todo, ...s.board.cols.doing].map((c) => c.id).sort();
    if (survivors.join(",") !== "1,2,3,4,5")
      return fail(`other columns damaged, ids now ${survivors.join(",")}`);
    return ok;
  },

  // ---------- wiki ----------
  "wiki-lisbon-year": (_s, t) => answerHas(t, ["2021"]),
  // The correct answer is the bare number — models phrase it endlessly
  // ("maximum of 5", "…is **5**", "five days"). Word-boundary match the
  // digit; the page's decoys (25 days PTO, $1,500 budget) can't produce a
  // standalone 5 token.
  "wiki-pto-carryover": (_s, t) =>
    /\b(5|five)\b/.test(norm(t)) ? ok : fail("final answer lacks a standalone 5/five"),
  "wiki-vp-product": (_s, t) => answerHas(t, ["atlas", "2019"]),

  // ---------- forms ----------
  "forms-register-growth": (s) => {
    const sub = (s.forms.submissions as any[]).find((x) => String(x.email) === "jordan.vega@acme.example");
    if (!sub) return fail("no submission for jordan.vega@acme.example");
    if (String(sub.first).toLowerCase() !== "jordan" || String(sub.last).toLowerCase() !== "vega")
      return fail(`name ${sub.first} ${sub.last}`);
    if (!String(sub.company).toLowerCase().includes("acme")) return fail(`company "${sub.company}"`);
    if (sub.teamSize !== "11-50") return fail(`teamSize "${sub.teamSize}"`);
    if (sub.plan !== "Growth") return fail(`plan "${sub.plan}"`);
    if (sub.start !== "2026-03-15") return fail(`start "${sub.start}"`);
    const addons = (sub.addons as string[]) ?? [];
    if (addons.length !== 1 || addons[0] !== "SSO") return fail(`addons ${JSON.stringify(addons)}, want [SSO]`);
    return ok;
  },

  "forms-date-retry": (s) => {
    const sub = (s.forms.submissions as any[]).find((x) => String(x.email) === "casey.lund@brightpath.example");
    if (!sub) return fail("no submission for casey.lund@brightpath.example");
    if (sub.plan !== "Starter") return fail(`plan "${sub.plan}"`);
    if (sub.start !== "2026-02-01") return fail(`start "${sub.start}", want the 2026-02-01 fallback`);
    if (((sub.addons as string[]) ?? []).length) return fail(`addons ${JSON.stringify(sub.addons)}, want none`);
    return ok;
  },

  "forms-quote": (_s, t) => answerAny(t, ["$89", "89 per", "89/user", "89 dollars"]),

  // ---------- admin ----------
  "admin-deactivate": (s) => {
    const u = s.admin.users.find((x) => x.id === 8)!;
    if (u.active) return fail("Hugo Lam is still active");
    if (u.role !== "Admin") return fail(`role changed to "${u.role}"`);
    // No collateral damage: everyone else keeps their seeded active flag.
    const seededActive = (id: number) => ((id - 1) * 13 + 2) % 5 !== 0;
    const damaged = s.admin.users.filter((x) => x.id !== 8 && x.active !== seededActive(x.id));
    if (damaged.length) return fail(`other users' active flags changed: ${damaged.map((d) => d.id).join(",")}`);
    return ok;
  },

  "admin-role-change": (s) => {
    const u = s.admin.users.find((x) => x.id === 24)!;
    if (u.role !== "Editor") return fail(`Xan Price role "${u.role}", want Editor`);
    if (!u.active) return fail("active flag was flipped");
    return ok;
  },

  "admin-count-viewers": (_s, t) => answerHas(t, ["10"]),
  "admin-newest-user": (_s, t) => answerHas(t, ["jack bloom", "2026-12-27"]),
};
