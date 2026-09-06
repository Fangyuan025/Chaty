// Browser-only IPC mock so the UI can be developed and visually verified with
// `vite dev` in a plain browser — no Tauri build needed. Installed by main.tsx
// ONLY when running in dev mode outside a Tauri webview; never ships.
//
// Fixtures are deliberately rich: they exercise every surface we polish
// (markdown showcase, thinking, code-agent steps with diffs/plan, models,
// hardware) so screenshots represent real-world density.

import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";

const now = Date.now();

const MODEL = {
  name: "Qwen3.6-35B-A3B-Q4_K_P.gguf",
  path: "/models/Qwen3.6-35B-A3B-Q4_K_P.gguf",
  backend: "llama.cpp",
  loaded: true,
  arch: "qwen3moe",
  sizeMb: 21800,
  paramsB: 34.7,
  nCtxTrain: 262144,
  nCtx: 40960,
  nLayer: 48,
  // The preview exercises the enabled half of the speculative-decoding switch;
  // the disabled half is what every model without a head shows.
  speculative: true,
  speculativeOn: true,
  gpuLayers: 48,
  gpuName: "Apple M4 Pro",
  modelName: "Qwen3.6 35B A3B",
  quant: "Q4_K_P",
  supportsThinking: true,
  thinkSwitch: false,
  multimodal: true,
  visionReady: true,
  mmproj: "/models/Qwen3.6-35B-A3B/mmproj-F16.gguf",
};

const CONVERSATIONS = [
  { id: "c1", title: "Explain Rust lifetimes", modelPath: MODEL.path, createdAt: now - 86400e3, updatedAt: now - 3600e3, pinned: true },
  { id: "c2", title: "Weekend trip around Montréal", modelPath: MODEL.path, createdAt: now - 2 * 86400e3, updatedAt: now - 7200e3, pinned: false },
  { id: "c3", title: "SQL window function examples", modelPath: MODEL.path, createdAt: now - 3 * 86400e3, updatedAt: now - 86400e3, pinned: false },
  { id: "c4", title: "Notes on the Q3 roadmap PDF", modelPath: MODEL.path, createdAt: now - 4 * 86400e3, updatedAt: now - 2 * 86400e3, pinned: false },
  { id: "c5", title: "KaTeX & table rendering", modelPath: MODEL.path, createdAt: now - 5 * 86400e3, updatedAt: now - 3 * 86400e3, pinned: false },
];

const SHOWCASE_MESSAGES = [
  { id: "m1", role: "user", content: "Explain Rust lifetimes with a code example and a comparison table, and throw in a formula." },
  {
    id: "m2",
    role: "assistant",
    content: `<think>They want code + a table + a formula. I'll use the classic borrow-checker example, a comparison table, and a KaTeX formula for the borrowing rule.</think>A lifetime is essentially **a name for the region a reference is valid**. Here's the classic example:

\`\`\`rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

fn main() {
    let s1 = String::from("long string is long");
    let result;
    {
        let s2 = String::from("xyz");
        result = longest(s1.as_str(), s2.as_str());
        println!("the longest is {result}");
    } // s2 is dropped here
}
\`\`\`

The key ideas at a glance:

| Concept | Meaning | Compiler behavior |
| --- | --- | --- |
| \`'a\` | Generic lifetime parameter | Takes the **shorter** of the two inputs |
| Dangling ref | Reference outlives its value | Rejected at compile time |
| NLL | Non-lexical lifetimes | A reference ends at its last use |

The borrowing rule can be summarized as: at any instant there are either $n \\ge 0$ shared borrows, or exactly $1$ mutable borrow:

$$\\text{borrows}(t) \\in \\{\\,(n, 0) : n \\ge 0\\,\\} \\cup \\{(0, 1)\\}$$

> In one line: **lifetimes change nothing at runtime — they're a proof handed to the compiler.**

If you want to go deeper, start with \`std::mem::drop\` and the NLL RFC.`,
  },
  { id: "m3", role: "user", content: "Nice. So what does 'static actually mean?", images: ["/Users/demo/Pictures/whiteboard.png"] },
  {
    id: "m4",
    role: "assistant",
    content: `<think>Two senses of 'static: the lifetime of literals, and the trait-bound meaning.</think>\`'static\` has two commonly-confused uses:

1. **A reference that lives as long as the program** — e.g. a string literal \`&'static str\`, which lives in the binary's read-only section;
2. **A type that holds no non-'static references** (the \`T: 'static\` bound) — meaning "T owns everything it contains", which \`String\` and \`Vec<u8>\` both satisfy.

The second has nothing to do with "living forever" — it just means **fully owned**. That's why \`thread::spawn\` requires \`'static\`: the thread may outlive the stack frame that created it.`,
  },
];

/** A rich Code-mode session: plan, web tools, code tools, diff, final answer. */
const CODE_SESSION_MSGS = [
  {
    id: "u1",
    role: "user",
    text: "Match the upstream fix for the tokenizer overflow — check their repo, then patch ours and run the tests.",
    steps: [],
    checkpointId: 3,
  },
  {
    id: "a1",
    role: "assistant",
    text: "Matched the upstream fix in `lexer.rs`: empty input now returns early in `tokenize` instead of reading past the end. Both call sites updated atomically, and all 96 tests pass.",
    thinking: "First find the upstream fix on GitHub, read the exact change, then locate and patch our tokenizer to match. Small, contained edit.",
    plan: [
      { content: "Find the upstream fix on GitHub", status: "done" },
      { content: "Locate our tokenizer and apply it", status: "done" },
      { content: "Re-run the test suite", status: "done" },
    ],
    steps: [
      {
        id: "s1",
        call: { name: "web_search", args: { query: "tokenizer overflow fix", site: "github.com" } },
        status: "done",
        result: "[repo] parser-labs/lexer (★2.4k, Rust) — a fast incremental lexer\n[issue] Panic on empty input in tokenize() [closed, 6 comments]\n[code] parser-labs/lexer/src/lexer.rs — fn tokenize(src)",
      },
      {
        id: "s2",
        call: { name: "web_fetch", args: { url: "https://github.com/parser-labs/lexer/blob/main/src/lexer.rs" } },
        status: "done",
        result: "https://raw.githubusercontent.com/parser-labs/lexer/main/src/lexer.rs [text]\npub fn tokenize(src: &str) -> Vec<Token> {\n    if src.is_empty() { return Vec::new(); }  // guard added in #214\n    …",
      },
      {
        id: "s3",
        call: { name: "search_files", args: { query: "tokenize" } },
        status: "done",
        result: "file names (1):\n  src/parser.ts\n\nfile contents:\nsrc/parser.ts:41: export function tokenize(src: string) {",
      },
      {
        id: "s4",
        call: { name: "outline", args: { path: "src/parser.ts" } },
        status: "done",
        result: "   12  export interface Token\n   41  export function tokenize(src: string)\n   88  function scanIdent(src: string, pos: number)",
      },
      {
        id: "s5",
        call: { name: "edit_file", args: { path: "src/parser.ts", edits: [{}, {}] } },
        status: "done",
        result: "Edited src/parser.ts (applied all 2 edits)",
        diff: {
          path: "src/parser.ts",
          before: "export function tokenize(src: string) {\n  let pos = 0;\n  const first = src.charAt(pos);",
          after: "export function tokenize(src: string) {\n  if (src.length === 0) return [];\n  let pos = 0;\n  const first = src.charAt(pos);",
        },
      },
      {
        id: "s6",
        call: { name: "bash", args: { command: "npm test" } },
        status: "done",
        result: "Tests: 96 passed, 96 total\nTime: 4.1s\n[exit 0]",
      },
    ],
  },
];

// Live session store: sessions saved during a preview run (deletable).
const SAVED_CODE_SESSIONS: { id: string; title: string; workspace: string }[] = [];
const SAVED_CODE_BODIES = new Map<string, string>();

const CODE_SESSIONS = [
  { id: "cs1", title: "Match the upstream tokenizer fix", workspace: "/Users/dev/projects/parser-kit", updatedAt: now - 1800e3 },
  { id: "cs2", title: "Add a README to the project", workspace: "/Users/dev/projects/parser-kit", updatedAt: now - 86400e3 },
];

/** Command → canned response. Anything unlisted returns a benign default. */
// Session dir grants (out-of-workspace access) — lets the grant pipeline be
// exercised end-to-end in the browser preview.
const GRANTS: string[] = [];
const SECRET_DIR = "/Users/dev/secrets";

function handle(cmd: string, args: Record<string, unknown> | undefined): unknown {
  switch (cmd) {
    // ---- model / hardware ----
    case "get_model":
      return MODEL;
    case "list_models":
      return [
        { name: MODEL.name, path: MODEL.path, sizeMb: MODEL.sizeMb, format: "gguf" },
        { name: "chaty-qwen3.5-4b-design-v3-Q4_K_M.gguf", path: "/models/chaty-4b.gguf", sizeMb: 2600, format: "gguf" },
        { name: "Gemma-4-E4B-Q8.gguf", path: "/models/gemma4.gguf", sizeMb: 4900, mmproj: "/models/gemma4/mmproj-F16.gguf", format: "gguf", vision: true },
        { name: "Qwen3.5-2B-4bit-MLX", path: "/models/Qwen3.5-2B-4bit-MLX", sizeMb: 1600, format: "mlx", vision: true },
        { name: "Qwen3-4B-4bit-MLX", path: "/models/Qwen3-4B-4bit-MLX", sizeMb: 2200, format: "mlx" },
      ];
    case "image_thumb":
      return "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='320' height='220'><defs><linearGradient id='s' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='%23aee3f5'/><stop offset='1' stop-color='%23e8f6dd'/></linearGradient></defs><rect width='320' height='220' fill='url(%23s)'/><circle cx='250' cy='58' r='26' fill='%23ffd66e'/><path d='M0 160 L90 92 L150 150 L210 105 L320 175 L320 220 L0 220 Z' fill='%236fae7a'/><path d='M0 190 L70 140 L160 195 L320 150 L320 220 L0 220 Z' fill='%23477a54'/></svg>";
    case "get_hardware_info":
      return { cpu: "Apple M4 Pro (14 核)", cpuThreads: 14, ramMb: 49152, gpuBackend: "Metal", gpu: { name: "Apple M4 Pro", vramMb: 40200 } };
    case "get_gpu_usage":
      return { usedMb: 22300, totalMb: 40200 };
    case "check_update":
      return { available: false, current: "1.5.0", latest: "1.5.0" };

    // ---- conversations ----
    case "list_conversations":
      return CONVERSATIONS;
    case "get_messages":
      return SHOWCASE_MESSAGES;
    case "search_conversations":
      return [];
    case "save_conversation":
    case "save_message":
    case "replace_messages":
    case "rename_conversation":
    case "set_conversation_pinned":
    case "delete_conversation":
    case "clear_all_conversations":
      return null;

    case "data_stats":
      return { conversations: 5, messages: 48, codeSessions: 2, dbBytes: 2_400_000 };

    // ---- extended web tools (Code mode) ----
    case "site_search": {
      const site = String(args?.site ?? "");
      if (/bilibili/.test(site)) {
        return [
          { kind: "video", title: "Rust 编程语言入门教程（已完结）(729:37, 软件工艺师, 1672594 播放)", url: "https://www.bilibili.com/video/BV1hp4y1k7SV", snippet: "Rust 权威指南配套视频教程" },
        ];
      }
      return [
        { kind: "repo", title: "fastapi/fastapi (★85000, Python)", url: "https://github.com/fastapi/fastapi", snippet: "FastAPI framework, high performance, easy to learn" },
        { kind: "code", title: "github.com/fastapi/fastapi/fastapi/main.py", url: "https://github.com/fastapi/fastapi/blob/HEAD/fastapi/main.py#L1", snippet: "from fastapi import FastAPI" },
      ];
    }
    case "fetch_page_ex": {
      const u = String(args?.url ?? "");
      if (/youtube\.com|youtu\.be/.test(u)) {
        return {
          url: u, kind: "video", contentType: "video/youtube",
          title: "示例视频", truncated: false,
          text: "视频 (video): 示例视频\n频道: Demo · 时长: 3:21\n\n—— 字幕转写 (transcript, en) ——\n[0:01] mock transcript for the preview environment",
          links: [], images: [], bytes: null,
        };
      }
      return {
        url: u, kind: "markdown", contentType: "text/html",
        title: "示例页面", text: "# 示例\n\n预览环境的模拟页面内容。", truncated: false,
        links: [{ url: "https://example.com/sub", text: "子页面" }], images: [], bytes: null,
      };
    }
    case "agent_web_download":
      return `已下载 ${String(args?.path ?? "file")} (12345 字节, image/png)`;
    case "agent_multi_edit":
      return `已编辑 ${String(args?.path ?? "?")}(应用全部 ${Array.isArray(args?.edits) ? (args.edits as unknown[]).length : 0} 处修改)`;
    case "agent_outline":
      return "    3  export function parse(s: string) {\n   12  class Lexer {\n   18    advance() {";

    // ---- knowledge base ----
    case "rag_status":
      return { modelReady: true, docs: 12, chunks: 486 };
    case "rag_clear_all":
      return null;
    case "rag_list_documents":
      return [
        { id: 1, name: "Product requirements v3.pdf", chunks: 84, enabled: true },
        { id: 2, name: "architecture-notes.md", chunks: 42, enabled: true },
        { id: 3, name: "Q2 financials.xlsx", chunks: 18, enabled: false },
      ];

    // ---- code mode ----
    case "code_session_list":
      return [...SAVED_CODE_SESSIONS, ...CODE_SESSIONS];
    case "code_session_load": {
      const hit = SAVED_CODE_SESSIONS.find((s) => s.id === String(args?.id ?? ""));
      if (hit) return SAVED_CODE_BODIES.get(hit.id) ?? "[]";
      return JSON.stringify(CODE_SESSION_MSGS);
    }
    case "code_session_save": {
      // Real save/delete semantics so delete-while-running is drivable in
      // the preview (the fixtures alone made every session immortal).
      const id = String(args?.id ?? "");
      const i = SAVED_CODE_SESSIONS.findIndex((s) => s.id === id);
      const meta = { id, title: String(args?.title ?? "session"), workspace: String(args?.workspace ?? "") };
      if (i >= 0) SAVED_CODE_SESSIONS[i] = meta;
      else SAVED_CODE_SESSIONS.unshift(meta);
      SAVED_CODE_BODIES.set(id, String(args?.data ?? "[]"));
      return null;
    }
    case "code_session_delete": {
      const id = String(args?.id ?? "");
      const i = SAVED_CODE_SESSIONS.findIndex((s) => s.id === id);
      if (i >= 0) SAVED_CODE_SESSIONS.splice(i, 1);
      SAVED_CODE_BODIES.delete(id);
      return null;
    }
    case "agent_grant_dir": {
      const d = String(args?.path ?? "");
      if (!GRANTS.includes(d)) GRANTS.push(d);
      return d;
    }
    case "agent_revoke_dir": {
      const d = String(args?.path ?? "");
      const i = GRANTS.indexOf(d);
      if (i >= 0) GRANTS.splice(i, 1);
      return null;
    }
    case "agent_list_grants":
      return [...GRANTS];
    case "agent_clear_grants":
      GRANTS.length = 0;
      return null;
    case "agent_bash":
      return { stdout: "(preview) cowsay installed", stderr: "", code: 0, timedOut: false };
    case "agent_read_file": {
      const path = String(args?.path ?? "");
      if (path.startsWith(SECRET_DIR)) {
        if (GRANTS.some((g) => path.startsWith(g))) {
          return "外部文件内容:数据集清单 v3 (outside-content: dataset manifest v3)";
        }
        // Same marker protocol as the Rust backend.
        throw `NEED_DIR_GRANT\t${SECRET_DIR}\t路径在工作区外，需要用户授权: ${path}`;
      }
      return "// mock file contents\nexport function tokenize(src: string) {}\n";
    }
    case "agent_get_workspace":
      return "/Users/dev/projects/parser-kit";
    case "agent_set_workspace":
      return String(args?.path ?? "");
    case "agent_bg_list":
    case "agent_bg_reap":
      return [];
    case "agent_checkpoint_begin":
      return 1;
    case "agent_list_files":
      return ["src/parser.ts", "src/index.ts", "src/parser.test.ts", "package.json", "README.md"];

    // ---- generation: a long prefill (progress ring 0→100%) then a short
    // streamed answer, so chat/Code streaming UI is testable in the browser ----
    case "generate": {
      const ch = args?.onEvent as { onmessage?: (ev: unknown) => void } | undefined;
      const emit = (ev: unknown) => ch?.onmessage?.(ev);
      const total = 6144;
      // Code-agent script: typing a task containing 外部文件 makes the mock
      // model read an out-of-workspace file, driving the dir-grant pipeline
      // (marker error → approval card → grant → retry) end-to-end in preview.
      const req = args?.request as { messages?: { role: string; content: string }[] } | undefined;
      const msgs = req?.messages ?? [];
      const last = msgs[msgs.length - 1]?.content ?? "";
      // Title-generation requests get a clean short title (never a tool_call).
      const sysAll = msgs.filter((m) => m.role === "system").map((m) => m.content).join("\n");
      const isTitleReq = /12个汉字|short chat title|作为对话标题/.test(sysAll);
      const wantsOutside = msgs.some((m) => m.role === "user" && (m.content.includes("外部文件") || /dataset manifest/i.test(m.content)));
      const outsideEn = msgs.some((m) => m.role === "user" && /dataset manifest/i.test(m.content));
      // Typing a task with 安装/sudo makes the mock model emit a sudo bash call,
      // driving the high-risk sudo approval dialog end-to-end in preview.
      const wantsSudo = msgs.some((m) => m.role === "user" && m.content.includes("sudo"));
      // Asking about the date makes the mock model echo the current-date line
      // injected into the Code system prompt — proves the agent is grounded.
      const wantsDate = msgs.some((m) => m.role === "user" && /日期|今天|date|today/i.test(m.content));
      const sysMsg = msgs.find((m) => m.role === "system")?.content ?? "";
      const dateEcho = (sysMsg.match(/当前日期时间:([^\n]+)/) || sysMsg.match(/Current date & time: ([^\n]+)/) || [])[1];
      const fast = wantsOutside || wantsSudo || wantsDate;
      let reply = "收到。这是浏览器预览的模拟输出 — the mock stream after a simulated prompt-processing phase.";
      if (isTitleReq) {
        reply = /dataset manifest/i.test(last) ? "Summarize dataset manifest" : "读取数据集清单";
      } else if (wantsDate && dateEcho) {
        reply = `根据系统提示,当前日期时间是:${dateEcho}`;
      } else if (wantsOutside) {
        if (last.includes("<tool_result")) {
          reply = last.includes("outside-content")
            ? (outsideEn
                ? "Read the external file — **dataset manifest v3**: 3 datasets, 12,400 files, last updated 2026-06. The granted folder shows as a chip in the header; revoke it anytime."
                : "已读取外部文件:数据集清单 v3。授权目录已在顶部显示,可随时取消。")
            : (outsideEn
                ? "The user denied access to that folder — continuing with workspace files only."
                : "用户拒绝了该目录的访问,我改用工作区内的资料继续。");
        } else {
          reply = '<tool_call>{"name":"read_file","arguments":{"path":"/Users/dev/secrets/manifest.txt"}}</tool_call>';
        }
      } else if (wantsSudo) {
        reply = last.includes("<tool_result")
          ? (last.includes("denied") || last.includes("拒绝")
              ? "用户拒绝了 sudo 命令,我改用无需管理员权限的方式。"
              : "命令已执行完成。")
          : '<tool_call>{"name":"bash","arguments":{"command":"sudo apt-get install -y cowsay"}}</tool_call>';
      }
      return (async () => {
        emit({ type: "started" });
        for (let done = 0; done <= total; done += 512) {
          emit({ type: "prefill", processed: Math.min(done, total), total });
          await new Promise((r) => setTimeout(r, fast ? 15 : 140));
        }
        for (const piece of reply.match(/.{1,6}/g) ?? []) {
          emit({ type: "token", text: piece });
          await new Promise((r) => setTimeout(r, 30));
        }
        emit({
          type: "done",
          stats: { promptTokens: total, completionTokens: 64, tokensPerSecond: 42.5, stopReason: "eos" },
        });
      })();
    }

    // ---- misc ----
    case "set_tray_language":
    case "open_data_dir":
    case "open_models_dir":
    case "open_external":
    case "browser_set_headless":
      return null;
    case "cancel_generation":
    case "cancel_download":
      return null;
    case "agent_dl_list":
    case "agent_dl_reap":
      return [];
    // Browser preview approximation of the native webview zoom.
    case "set_ui_zoom": {
      const f = Number(args?.factor ?? 1);
      (document.body.style as CSSStyleDeclaration & { zoom: string }).zoom = f === 1 ? "" : String(f);
      return null;
    }
    // 0.4s of silence so the Settings voice preview is testable in the browser.
    case "synthesize": {
      const silence = new Uint8Array(4 * 9600);
      let bin = "";
      for (let i = 0; i < silence.length; i += 8192) {
        bin += String.fromCharCode(...silence.subarray(i, i + 8192));
      }
      return { audio: btoa(bin), sampleRate: 24000 };
    }

    case "hf_author_avatar": {
      const av: Record<string, string> = {
        Qwen: "https://cdn-avatars.huggingface.co/v1/production/uploads/6215ca5692c0ecfba9186921/hrRM50-6XcdWgg2AKpENG.jpeg",
        google: "https://cdn-avatars.huggingface.co/v1/production/uploads/5dd96eb166059660ed1ee413/WtA3YYitedOr9n02eHfJe.png",
        "mlx-community": "https://cdn-avatars.huggingface.co/v1/production/uploads/623c830997ddced06d78699b/3qTjC7d3YFCJTwpxd2noq.png",
      };
      return av[(args?.author as string) ?? ""] ?? null;
    }
    case "hf_search":
      return [
        { id: "Qwen/Qwen3-4B-GGUF", name: "Qwen3-4B-GGUF", author: "Qwen", downloads: 2512124, likes: 2710, updatedAt: new Date(Date.now() - 38 * 864e5).toISOString(), vision: false, paramsB: 4 },
        { id: "google/gemma-4-12b-qat-GGUF", name: "gemma-4-12b-qat-GGUF", author: "google", downloads: 901906, likes: 1074, updatedAt: new Date(Date.now() - 40 * 864e5).toISOString(), vision: true, paramsB: 12 },
        { id: "mlx-community/Qwen3.5-2B-4bit", name: "Qwen3.5-2B-4bit", author: "mlx-community", downloads: 315434, likes: 101, updatedAt: new Date(Date.now() - 5 * 864e5).toISOString(), vision: true, paramsB: 2 },
      ];
    case "hf_model_detail":
      return {
        id: (args?.repo as string) ?? "Qwen/Qwen3-4B-GGUF",
        format: "gguf",
        vision: true,
        paramsB: 12,
        arch: "gemma4",
        quants: [
          { label: "Q4_0", size: 7.15 * 2 ** 30, files: ["gemma-4-12b-qat-Q4_0.gguf"] },
          { label: "Q8_0", size: 12.8 * 2 ** 30, files: ["gemma-4-12b-qat-Q8_0.gguf"] },
        ],
        mmproj: "mmproj-F16.gguf",
        mmprojSize: 812 * 2 ** 20,
        readme: '<p align="center"> <img src="assets/banner.png" alt="banner" width="100%"/> </p>\n\n# Gemma 4 12B QAT\n\n<b>V2.0 is available</b> — Gemma 4 12B QAT is the <em>Quantization-Aware Training</em> version of Gemma 4 12B. It aims to keep quality close to bfloat16 while using much less memory.\n\n- Text and image input\n- 128K context\n\n<details><summary>Benchmarks</summary>MMLU 78.3 · GSM8K 91.2</details>\n\n![chart](./assets/chart.png)',
        totalRamMb: 49152,
      };
    case "list_hf_ggufs":
      throw "该仓库没有 .gguf 文件";
    case "list_hf_mlx":
      return { name: "Qwen3-4B-4bit-MLX", files: 9, totalSize: 2306867200 };

    default:
      console.warn(`[devMock] unhandled command: ${cmd}`, args);
      return null;
  }
}

export function installDevMock() {
  // plugin-os `platform()` reads an injected global; fake macOS.
  (window as unknown as Record<string, unknown>).__TAURI_OS_PLUGIN_INTERNALS__ = {
    platform: "macos",
    os_type: "macos",
    family: "unix",
    version: "15.0",
    arch: "aarch64",
    exe_extension: "",
    eol: "\n",
  };
  mockWindows("main");
  mockIPC((cmd, args) => {
    // Tauri plugins arrive as "plugin:name|method".
    if (cmd.startsWith("plugin:")) {
      if (cmd === "plugin:os|platform") return "macos";
      return null;
    }
    return handle(cmd, args as Record<string, unknown> | undefined);
  });
  // getCurrentWebview() wants webview metadata that mockWindows doesn't set.
  const internals = (window as unknown as Record<string, any>).__TAURI_INTERNALS__;
  if (internals?.metadata) {
    internals.metadata.currentWebview ??= { label: "main" };
    internals.metadata.webviews ??= [{ label: "main" }];
  }
  console.info("[devMock] Tauri IPC mocked — browser preview mode");
}
