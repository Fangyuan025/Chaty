/** ChatyWebapp-Bench scenarios: the four audited webapp failure modes, each a
 *  self-contained mini-project (no npm install — python3 http.server keeps it
 *  offline and deterministic). Grading is objective: file state, server
 *  liveness, and step-log facts — never the model's self-report. */

export interface WebappScenario {
  id: string;
  /** What the audit pain point is, for the report. */
  probes: string;
  instruction: string;
  files: Record<string, string>;
  /** -1 = a fresh port is assigned per run ({PORT} token); 0 = no server. */
  port: number;
  /** File-state checks: every [file, mustContain] pair must hold at the end. */
  expectFiles?: [string, string][];
  /** Negative checks: [file, mustNOTContain] — catches "explained but never
   *  fixed" and needles the buggy original would also match. */
  expectFilesAbsent?: [string, string][];
  /** Any-of checks: [fileOrPipeJoined, [candidates]] — at least one candidate
   *  must appear in the concatenation of the |-separated files (the button
   *  wiring legitimately lives in either the JS or the HTML). */
  expectFilesAny?: [string, string[]][];
  /** A browser_click must have happened (the change is behavioral). */
  expectClicked?: boolean;
  /** The final answer must contain this string. */
  expectAnswer?: string;
  /** The dev server must still be reachable when the turn ends. */
  expectServerAlive?: boolean;
  /** A browser action must have happened AFTER the last file edit. */
  expectBrowserAfterEdit?: boolean;
}

const pkg = () =>
  JSON.stringify({ name: "bench-app", scripts: { dev: "python3 -u -m http.server {PORT}" } }, null, 2);

export const SCENARIOS: WebappScenario[] = [
  {
    id: "server-start",
    probes: "痛点④ dev server 前台跑 → 停滞/被杀",
    instruction:
      "Start this project's dev server (see the `dev` script in package.json). Then tell me the exact local URL it serves and the page title of the homepage. Leave the server running.",
    port: -1,
    files: {
      "package.json": pkg(),
      "index.html": "<!doctype html><html><head><title>Bench Home {PORT}</title></head><body><h1>home</h1></body></html>",
    },
    expectAnswer: "{PORT}",
    expectServerAlive: true,
  },
  {
    id: "console-fix",
    probes: "痛点③ 不看 console → 修不到点上",
    instruction:
      "Start the dev server (script `dev`) and open http://localhost:{PORT}/ in the browser. The page should show the text 'Hello, Chaty!' inside #greeting, but it shows '...' instead. Find the actual bug, fix it, and verify in the browser that the greeting renders before you answer.",
    port: -1,
    files: {
      "package.json": pkg(),
      "index.html":
        '<!doctype html><html><head><title>Greet</title></head><body><div id="greeting">...</div><script src="app.js"></script></body></html>',
      // ReferenceError at load: getGreting is not defined (typo'd call site).
      "app.js":
        'function getGreeting() {\n  return "Hello, Chaty!";\n}\ndocument.getElementById("greeting").textContent = getGreting();\n',
    },
    expectFilesAbsent: [["app.js", "getGreting"]],
    expectBrowserAfterEdit: true,
  },
  {
    id: "todo-follow",
    probes: "痛点① todo 成摆设 → 三项只交一两项",
    instruction:
      "Make ALL THREE of these changes to this static site — plan them with update_plan first and keep the statuses updated as you go: (1) change the page title to exactly 'Dashboard v2'; (2) add a footer element containing '© 2026 Chaty' at the end of the body; (3) link style.css in the <head> with a <link> tag. All three are required.",
    port: 0,
    files: {
      "index.html":
        "<!doctype html><html><head><title>Old Dashboard</title></head><body><h1>Numbers</h1><p>content</p></body></html>",
      "style.css": "h1 { color: #333; }\n",
    },
    expectFiles: [
      ["index.html", "Dashboard v2"],
      ["index.html", "© 2026 Chaty"],
      ["index.html", "style.css"],
    ],
  },
  {
    id: "ship-verified",
    probes: "痛点② 交付草率 → 不做浏览器走查",
    instruction:
      "Start the dev server (script `dev`, serves http://localhost:{PORT}/). Then make two changes: (1) in style.css set the h1 color to #224466; (2) the 'Add' button is currently a no-op — wire it up so clicking it appends a new <li>item</li> to the #list. Verify your changes in the real page (actually exercise the button in the browser), then answer.",
    port: -1,
    files: {
      "package.json": pkg(),
      "index.html":
        '<!doctype html><html><head><title>List</title><link rel="stylesheet" href="style.css"></head><body><h1>My List</h1><button id="add">Add</button><ul id="list"></ul><script src="app.js"></script></body></html>',
      "style.css": "h1 { color: #999999; }\n",
      // The handler exists but is never attached — the audited "looks done in
      // the code, dead in the page" trap.
      "app.js":
        'function addItem() {\n  const li = document.createElement("li");\n  li.textContent = "item";\n  document.getElementById("list").appendChild(li);\n}\n',
    },
    expectFiles: [["style.css", "#224466"]],
    expectFilesAny: [["app.js|index.html", ["addEventListener", "onclick"]]],
    expectClicked: true,
    expectBrowserAfterEdit: true,
  },
];
