/**
 * ChatyWeb-Bench fixture server — deterministic local mini-web.
 *
 * Serves six self-contained single-file apps (sites/*.html) plus a JSON state
 * API. The server is the single source of truth: every mutation an app makes
 * goes through POST /api/..., so graders only ever need GET /api/state plus
 * the agent's final message. POST /api/reset restores the seeded world.
 *
 * Zero dependencies; run standalone with `npx tsx server.mts` or import
 * { startServer, resetState, getState } from the oracle/runner.
 */
import http from "node:http";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIR = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_PORT = 8763;

// ---------- seeds ----------

export const PRODUCTS = [
  { id: "p01", name: "Aurora Wireless Mouse", cat: "Accessories", price: 24.99, stock: 12 },
  { id: "p02", name: "Borealis Mechanical Keyboard", cat: "Accessories", price: 89.0, stock: 7 },
  { id: "p03", name: "Cinder USB-C Cable 1m", cat: "Cables", price: 6.5, stock: 40 },
  { id: "p04", name: "Cinder USB-C Cable 2m", cat: "Cables", price: 8.99, stock: 35 },
  { id: "p05", name: "Drift HDMI Cable 4K", cat: "Cables", price: 12.0, stock: 22 },
  { id: "p06", name: "Ember Compact Keyboard", cat: "Accessories", price: 45.5, stock: 9 },
  { id: "p07", name: "Flux Studio Headphones", cat: "Audio", price: 129.99, stock: 5 },
  { id: "p08", name: "Glacier Earbuds", cat: "Audio", price: 39.99, stock: 18 },
  { id: "p09", name: "Harbor Desk Mat XL", cat: "Accessories", price: 19.0, stock: 25 },
  { id: "p10", name: "Ion Portable Speaker", cat: "Audio", price: 59.0, stock: 11 },
  { id: "p11", name: "Juniper Webcam 1080p", cat: "Video", price: 49.99, stock: 14 },
  { id: "p12", name: "Krypton Ring Light", cat: "Video", price: 27.5, stock: 16 },
  { id: "p13", name: "Lumen Monitor 27in", cat: "Displays", price: 249.0, stock: 4 },
  { id: "p14", name: "Mesa Laptop Stand", cat: "Accessories", price: 32.0, stock: 20 },
  { id: "p15", name: "Nimbus Ethernet Cable 5m", cat: "Cables", price: 9.75, stock: 30 },
  { id: "p16", name: "Onyx Mic Arm", cat: "Audio", price: 34.0, stock: 8 },
  { id: "p17", name: "Pico Condenser Mic", cat: "Audio", price: 74.5, stock: 6 },
  { id: "p18", name: "Quartz Monitor 24in", cat: "Displays", price: 179.0, stock: 6 },
  { id: "p19", name: "Ridge Phone Dock", cat: "Accessories", price: 15.25, stock: 28 },
  { id: "p20", name: "Slate Drawing Tablet", cat: "Accessories", price: 99.0, stock: 5 },
  { id: "p21", name: "Terra AUX Cable", cat: "Cables", price: 4.99, stock: 50 },
  { id: "p22", name: "Umbra Privacy Screen 27in", cat: "Displays", price: 42.0, stock: 10 },
  { id: "p23", name: "Vista Capture Card", cat: "Video", price: 119.0, stock: 3 },
  { id: "p24", name: "Willow Numpad", cat: "Accessories", price: 21.5, stock: 13 },
];

function seedEmails() {
  return [
    { id: 1, from: "Acme Billing <billing@acme.example>", subject: "Invoice #INV-2093 — payment due", body: "Hello,\n\nInvoice INV-2093 for $1,240.00 is due on Friday. Reply to this message with your purchase order number so we can match the payment.\n\n— Acme Billing", read: false, archived: false, replies: [] as string[] },
    { id: 2, from: "The Daily Byte <news@dailybyte.example>", subject: "Newsletter: Rust 2.0 rumors, and more", body: "This week in tech: speculation about Rust 2.0, a new GPU generation, and our favorite keyboards.", read: true, archived: false, replies: [] },
    { id: 3, from: "BlueSky Travel <deals@bluesky.example>", subject: "Your spring getaway awaits", body: "Book any flight before March 31 and use discount code SKYFALL25 at checkout for 25% off your first booking.", read: false, archived: false, replies: [] },
    { id: 4, from: "Mira Chen <mira@nortonlabs.example>", subject: "Quarterly sync moved", body: "Heads up — the quarterly sync moved to Thursday 14:00 in the Redwood room. Agenda unchanged.", read: true, archived: false, replies: [] },
    { id: 5, from: "Procurement <procurement@yourco.example>", subject: "PO issued for Acme services", body: "FYI: purchase order PO-7741 has been issued to Acme Corp for Q3 consulting services. Reference it in any billing correspondence.", read: false, archived: false, replies: [] },
    { id: 6, from: "The Daily Byte <news@dailybyte.example>", subject: "Newsletter: The week in AI", body: "Local models keep getting better. Also: a deep dive into quantization formats.", read: true, archived: false, replies: [] },
    { id: 7, from: "GitStream <noreply@gitstream.example>", subject: "Your weekly repo digest", body: "12 commits, 3 merged PRs, 1 new contributor this week on chaty/desktop.", read: true, archived: false, replies: [] },
    { id: 8, from: "Dana Whitfield <dana@yourco.example>", subject: "Offsite headcount", body: "Can you confirm how many from your team are joining the June offsite? Need numbers by Wednesday.", read: false, archived: false, replies: [] },
    { id: 9, from: "The Daily Byte <news@dailybyte.example>", subject: "Newsletter: Special edition", body: "Our annual hardware guide is out.", read: false, archived: false, replies: [] },
    { id: 10, from: "CloudMetrics <alerts@cloudmetrics.example>", subject: "ALERT: staging CPU above 90%", body: "staging-worker-3 sustained >90% CPU for 15 minutes. Auto-scaled to 4 replicas.", read: true, archived: false, replies: [] },
    { id: 11, from: "Rene Okafor <rene@brightpath.example>", subject: "Partnership intro", body: "Great meeting you at the expo. Would love to explore a co-marketing pilot in Q3.", read: false, archived: false, replies: [] },
    { id: 12, from: "IT Helpdesk <it@yourco.example>", subject: "Password rotation reminder", body: "Company policy: workstation passwords rotate every 90 days. Yours expires next Monday.", read: true, archived: false, replies: [] },
    { id: 13, from: "Acme Support <support@acme.example>", subject: "Ticket #4482 resolved", body: "Your ticket about API rate limits has been resolved. Limits were raised to 600 req/min.", read: true, archived: false, replies: [] },
    { id: 14, from: "Festival Committee <events@yourco.example>", subject: "Summer festival volunteers", body: "We still need 4 volunteers for the summer festival booth. Sign-up closes Friday.", read: false, archived: false, replies: [] },
  ];
}

function seedBoard() {
  return {
    nextId: 9,
    cols: {
      todo: [
        { id: 1, title: "Write Q3 roadmap draft", assignee: "Mira" },
        { id: 2, title: "Fix login bug", assignee: "Sam" },
        { id: 3, title: "Update onboarding docs", assignee: "Dana" },
      ],
      doing: [
        { id: 4, title: "Migrate CI to new runners", assignee: "Sam" },
        { id: 5, title: "Design settings page", assignee: "Mira" },
      ],
      done: [
        { id: 6, title: "Ship v2.3.1 hotfix", assignee: "Sam" },
        { id: 7, title: "Renew TLS certificates", assignee: "Sam" },
        { id: 8, title: "Archive stale branches", assignee: "Mira" },
      ],
    },
  };
}

const FIRST = ["Ava", "Ben", "Cora", "Dev", "Elle", "Finn", "Gia", "Hugo", "Iris", "Jude", "Kai", "Lena", "Milo", "Nora", "Omar", "Pia", "Quinn", "Rhea", "Seth", "Tara", "Uma", "Vik", "Wren", "Xan", "Yara", "Zane", "Ada", "Bram", "Cleo", "Dion", "Esme", "Ford", "Gwen", "Hal", "Ines", "Jack", "Kira"];
const LAST = ["Stone", "Reyes", "Park", "Novak", "Idris", "Frost", "Vega", "Lam", "Osei", "Brand", "Cruz", "Dietz", "Ellis", "Ford", "Gale", "Haas", "Ibarra", "Joyce", "Katz", "Lund", "Marsh", "Nash", "Odell", "Price", "Quist", "Rowe", "Shaw", "Tate", "Ubaldo", "Voss", "Witt", "Xiong", "Yates", "Zorn", "Ames", "Bloom", "Corr"];
const ROLES = ["Admin", "Manager", "Editor", "Viewer"];

function seedUsers() {
  // Deterministic 37 rows; joined dates spread across 2024-2026.
  return FIRST.map((f, i) => {
    const l = LAST[i];
    const role = ROLES[(i * 7 + 3) % 4];
    const yy = 2024 + ((i * 5 + 1) % 3);
    const mm = String(((i * 3 + 2) % 12) + 1).padStart(2, "0");
    const dd = String(((i * 11 + 5) % 28) + 1).padStart(2, "0");
    return {
      id: i + 1,
      name: `${f} ${l}`,
      email: `${f.toLowerCase()}.${l.toLowerCase()}@nimbusworks.example`,
      role,
      active: (i * 13 + 2) % 5 !== 0,
      joined: `${yy}-${mm}-${dd}`,
    };
  });
}

export function seedState() {
  return {
    shop: { cart: [] as { id: string; qty: number }[], orders: [] as Json[] },
    inbox: { emails: seedEmails(), sent: [] as Json[] },
    board: seedBoard(),
    forms: { submissions: [] as Json[] },
    admin: { users: seedUsers() },
  };
}

export type Json = Record<string, unknown> | unknown[] | string | number | boolean | null;
type State = ReturnType<typeof seedState>;

let state: State = seedState();
export const getState = (): State => state;
export const resetState = (): void => {
  state = seedState();
};

// ---------- API ----------

function api(method: string, route: string, body: Json): { code: number; data: Json } {
  const S = state;
  const b = (body ?? {}) as Record<string, any>;
  const ok = (data: Json = { ok: true }) => ({ code: 200, data });
  const bad = (error: string) => ({ code: 400, data: { error } });

  if (method === "GET" && route === "/api/state") return ok(S as unknown as Json);
  if (method === "POST" && route === "/api/reset") {
    resetState();
    return ok();
  }

  // shop
  if (method === "GET" && route === "/api/shop/data")
    return ok({ products: PRODUCTS, cart: S.shop.cart });
  if (method === "POST" && route === "/api/shop/cart/add") {
    const p = PRODUCTS.find((x) => x.id === b.id);
    if (!p) return bad("unknown product");
    const qty = Number(b.qty) || 1;
    const row = S.shop.cart.find((x) => x.id === b.id);
    if (row) row.qty += qty;
    else S.shop.cart.push({ id: b.id, qty });
    return ok({ cart: S.shop.cart });
  }
  if (method === "POST" && route === "/api/shop/cart/remove") {
    S.shop.cart = S.shop.cart.filter((x) => x.id !== b.id);
    return ok({ cart: S.shop.cart });
  }
  if (method === "POST" && route === "/api/shop/checkout") {
    if (!S.shop.cart.length) return bad("cart is empty");
    for (const k of ["name", "address", "shipping"])
      if (!String(b[k] ?? "").trim()) return bad(`missing ${k}`);
    if (!["standard", "express"].includes(b.shipping)) return bad("shipping must be standard or express");
    const items = S.shop.cart.map((c) => ({ ...c, price: PRODUCTS.find((p) => p.id === c.id)!.price }));
    const total = Math.round(items.reduce((s, i) => s + i.price * i.qty, 0) * 100) / 100;
    S.shop.orders.push({ items, total, name: b.name, address: b.address, shipping: b.shipping });
    S.shop.cart = [];
    return ok({ ordered: true, total });
  }

  // inbox
  if (method === "GET" && route === "/api/inbox/data") return ok({ emails: S.inbox.emails });
  if (method === "POST" && route === "/api/inbox/open") {
    const e = S.inbox.emails.find((x) => x.id === Number(b.id));
    if (!e) return bad("no such email");
    e.read = true;
    return ok({ email: e });
  }
  if (method === "POST" && route === "/api/inbox/reply") {
    const e = S.inbox.emails.find((x) => x.id === Number(b.id));
    if (!e) return bad("no such email");
    if (!String(b.body ?? "").trim()) return bad("empty reply");
    e.replies.push(String(b.body));
    return ok();
  }
  if (method === "POST" && route === "/api/inbox/archive") {
    const e = S.inbox.emails.find((x) => x.id === Number(b.id));
    if (!e) return bad("no such email");
    e.archived = true;
    return ok();
  }
  if (method === "POST" && route === "/api/inbox/compose") {
    for (const k of ["to", "subject", "body"]) if (!String(b[k] ?? "").trim()) return bad(`missing ${k}`);
    S.inbox.sent.push({ to: b.to, subject: b.subject, body: b.body });
    return ok();
  }

  // board
  if (method === "GET" && route === "/api/board/data") return ok({ cols: S.board.cols });
  if (method === "POST" && route === "/api/board/add") {
    const col = String(b.col ?? "");
    if (!(col in S.board.cols)) return bad("col must be todo|doing|done");
    if (!String(b.title ?? "").trim()) return bad("missing title");
    S.board.cols[col as "todo"].push({ id: S.board.nextId++, title: String(b.title), assignee: String(b.assignee ?? "") });
    return ok({ cols: S.board.cols });
  }
  if (method === "POST" && route === "/api/board/move") {
    const order = ["todo", "doing", "done"] as const;
    for (const c of order) {
      const i = S.board.cols[c].findIndex((x) => x.id === Number(b.id));
      if (i >= 0) {
        const to = order[order.indexOf(c) + (b.dir === "left" ? -1 : 1)];
        if (!to) return bad("cannot move further");
        S.board.cols[to].push(S.board.cols[c].splice(i, 1)[0]);
        return ok({ cols: S.board.cols });
      }
    }
    return bad("no such card");
  }
  if (method === "POST" && route === "/api/board/del") {
    for (const c of ["todo", "doing", "done"] as const)
      S.board.cols[c] = S.board.cols[c].filter((x) => x.id !== Number(b.id));
    return ok({ cols: S.board.cols });
  }

  // forms (reference "today" is fixed for determinism)
  if (method === "POST" && route === "/api/forms/submit") {
    const required = ["first", "last", "email", "company", "teamSize", "plan", "start"];
    for (const k of required) if (!String(b[k] ?? "").trim()) return bad(`missing ${k}`);
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(String(b.email))) return bad("invalid email format");
    if (b.terms !== true) return bad("terms must be accepted");
    if (String(b.start) <= "2026-01-15") return bad("start date must be after 2026-01-15");
    S.forms.submissions.push({
      first: b.first, last: b.last, email: b.email, company: b.company,
      teamSize: b.teamSize, plan: b.plan, start: b.start,
      addons: Array.isArray(b.addons) ? b.addons : [],
    });
    return ok({ submitted: true });
  }

  // admin
  if (method === "GET" && route === "/api/admin/data") return ok({ users: S.admin.users });
  if (method === "POST" && route === "/api/admin/update") {
    const u = S.admin.users.find((x) => x.id === Number(b.id));
    if (!u) return bad("no such user");
    if (b.role !== undefined) {
      if (!ROLES.includes(String(b.role))) return bad("bad role");
      u.role = String(b.role);
    }
    if (b.active !== undefined) u.active = Boolean(b.active);
    return ok({ user: u });
  }

  return { code: 404, data: { error: `no route: ${method} ${route}` } };
}

// ---------- http ----------

export function startServer(port = DEFAULT_PORT): Promise<http.Server> {
  const srv = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://127.0.0.1:${port}`);
    const route = url.pathname;
    if (route.startsWith("/api/")) {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        let body: Json = null;
        try {
          body = raw ? JSON.parse(raw) : null;
        } catch {
          /* fall through with null body */
        }
        const { code, data } = api(req.method ?? "GET", route, body);
        res.writeHead(code, { "Content-Type": "application/json" });
        res.end(JSON.stringify(data));
      });
      return;
    }
    const file = route === "/" ? "/index.html" : route;
    const p = path.join(DIR, "sites", path.basename(file));
    if (existsSync(p) && p.endsWith(".html")) {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(readFileSync(p));
      return;
    }
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("not found");
  });
  return new Promise((resolve, reject) => {
    srv.once("error", reject);
    srv.listen(port, "127.0.0.1", () => resolve(srv));
  });
}

// Standalone: `npx tsx server.mts [port]`
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const port = Number(process.argv[2]) || DEFAULT_PORT;
  startServer(port).then(() => console.log(`ChatyWeb-Bench fixtures on http://127.0.0.1:${port}/`));
}
