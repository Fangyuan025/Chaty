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
  gpuLayers: 48,
  gpuName: "Apple M4 Pro",
  modelName: "Qwen3.6 35B A3B",
  quant: "Q4_K_P",
  supportsThinking: true,
  thinkSwitch: false,
};

const CONVERSATIONS = [
  { id: "c1", title: "Rust 生命周期怎么理解", modelPath: MODEL.path, createdAt: now - 86400e3, updatedAt: now - 3600e3, pinned: true },
  { id: "c2", title: "周末旅行计划:蒙特利尔周边", modelPath: MODEL.path, createdAt: now - 2 * 86400e3, updatedAt: now - 7200e3, pinned: false },
  { id: "c3", title: "SQL 窗口函数示例", modelPath: MODEL.path, createdAt: now - 3 * 86400e3, updatedAt: now - 86400e3, pinned: false },
  { id: "c4", title: "读书笔记:置身事内", modelPath: MODEL.path, createdAt: now - 4 * 86400e3, updatedAt: now - 2 * 86400e3, pinned: false },
  { id: "c5", title: "KaTeX 与表格渲染测试", modelPath: MODEL.path, createdAt: now - 5 * 86400e3, updatedAt: now - 3 * 86400e3, pinned: false },
];

const SHOWCASE_MESSAGES = [
  { id: "m1", role: "user", content: "用一段代码和一张表格解释 Rust 的生命周期,顺便写个公式。" },
  {
    id: "m2",
    role: "assistant",
    content: `<think>用户想要代码 + 表格 + 公式的组合示例。我用一个借用检查的经典场景,配一张对比表,再放一个无关紧要但能展示 KaTeX 的公式。</think>生命周期本质上是**引用有效范围的名字**。看这个例子:

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
        println!("最长的是 {result}");
    } // s2 在这里被 drop
}
\`\`\`

关键点对比:

| 概念 | 含义 | 编译器行为 |
| --- | --- | --- |
| \`'a\` | 泛型生命周期参数 | 取两个输入中**较短**的那个 |
| 悬垂引用 | 引用超过了值的存活期 | 直接拒绝编译 |
| NLL | 非词法生命周期 | 引用最后一次使用后即失效 |

顺带一个公式,借用规则可以概括为:在任意时刻,要么有 $n \\ge 0$ 个不可变借用,要么恰有 $1$ 个可变借用:

$$\\text{borrows}(t) \\in \\{\\,(n, 0) : n \\ge 0\\,\\} \\cup \\{(0, 1)\\}$$

> 一句话:**生命周期不改变任何运行时行为,它只是给编译器的证明。**

想深入的话可以从 \`std::mem::drop\` 和 NLL 的 RFC 读起。`,
  },
  { id: "m3", role: "user", content: "不错。那 'static 是什么意思?" },
  {
    id: "m4",
    role: "assistant",
    content: `<think>简短回答 'static 的两种含义:字面量的生命周期和 trait bound 中的含义。</think>\`'static\` 有两个常被混淆的用法:

1. **引用活得和程序一样久** —— 比如字符串字面量 \`&'static str\`,它存在二进制的只读段里;
2. **类型不含任何非 'static 引用**(trait bound 里的 \`T: 'static\`)—— 表示"T 拥有它的一切",比如 \`String\`、\`Vec<u8>\` 都满足。

第二种和"活得久"无关,只是**所有权完整**的意思——这也是 \`thread::spawn\` 要求 \`'static\` 的原因:线程可能比创建它的栈帧活得久。`,
  },
];

/** A rich Code-mode session: plan, steps of every kind, diff, final answer. */
const CODE_SESSION_MSGS = [
  {
    id: "u1",
    role: "user",
    text: "跑一下测试,把失败的修了",
    steps: [],
    checkpointId: 3,
  },
  {
    id: "a1",
    role: "assistant",
    text: "已修复 `parser.ts` 的边界情况:空输入会在 `tokenize` 里提前返回,不再触发越界读取。全部 42 个测试通过。",
    thinking: "测试失败在 tokenize 的空字符串分支,补一个 guard 即可。改动很小,直接修。",
    plan: [
      { content: "运行测试套件,定位失败用例", status: "done" },
      { content: "修复 parser.ts 空输入越界", status: "done" },
      { content: "重跑测试确认全绿", status: "done" },
    ],
    steps: [
      {
        id: "s1",
        call: { name: "bash", args: { command: "npm test" } },
        status: "done",
        result: "FAIL src/parser.test.ts\n  ✕ tokenize handles empty input (3 ms)\n\nTests: 1 failed, 41 passed, 42 total\n[exit 1]",
        thinking: "先看失败的是哪个用例。",
      },
      {
        id: "s2",
        call: { name: "search_code", args: { query: "tokenize empty input" } },
        status: "done",
        result: "── src/parser.ts:41 ──\nexport function tokenize(src: string) {\n  let pos = 0;\n  // BUG: no empty check before charAt\n  ...",
      },
      {
        id: "s3",
        call: { name: "edit_file", args: { path: "src/parser.ts" } },
        status: "done",
        result: "已编辑 src/parser.ts(替换 1 处)",
        diff: {
          path: "src/parser.ts",
          before: "export function tokenize(src: string) {\n  let pos = 0;\n  const first = src.charAt(pos);",
          after: "export function tokenize(src: string) {\n  if (src.length === 0) return [];\n  let pos = 0;\n  const first = src.charAt(pos);",
        },
      },
      {
        id: "s4",
        call: { name: "bash", args: { command: "npm test" } },
        status: "done",
        result: "Tests: 42 passed, 42 total\nTime: 3.4s\n[exit 0]",
      },
      {
        id: "s5",
        call: { name: "bash", args: { command: "npm run lint" } },
        status: "error",
        result: "ERROR: 命令超时 (timed out)\n[exit -1]",
      },
    ],
  },
];

const CODE_SESSIONS = [
  { id: "cs1", title: "跑一下测试,把失败的修了", workspace: "/Users/dev/projects/parser-kit", updatedAt: now - 1800e3 },
  { id: "cs2", title: "给项目补一个 README", workspace: "/Users/dev/projects/parser-kit", updatedAt: now - 86400e3 },
];

/** Command → canned response. Anything unlisted returns a benign default. */
function handle(cmd: string, args: Record<string, unknown> | undefined): unknown {
  switch (cmd) {
    // ---- model / hardware ----
    case "get_model":
      return MODEL;
    case "list_models":
      return [
        { name: MODEL.name, path: MODEL.path, sizeMb: MODEL.sizeMb },
        { name: "chaty-qwen3.5-4b-design-v3-Q4_K_M.gguf", path: "/models/chaty-4b.gguf", sizeMb: 2600 },
        { name: "Gemma-4-E4B-Q8.gguf", path: "/models/gemma4.gguf", sizeMb: 4900 },
      ];
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

    // ---- knowledge base ----
    case "rag_status":
      return { modelReady: true, docs: 12, chunks: 486 };
    case "rag_clear_all":
      return null;
    case "rag_list_documents":
      return [
        { id: 1, name: "产品需求文档 v3.pdf", chunks: 84, enabled: true },
        { id: 2, name: "架构设计.md", chunks: 42, enabled: true },
        { id: 3, name: "季度财务摘要.xlsx", chunks: 18, enabled: false },
      ];

    // ---- code mode ----
    case "code_session_list":
      return CODE_SESSIONS;
    case "code_session_load":
      return JSON.stringify(CODE_SESSION_MSGS);
    case "code_session_save":
    case "code_session_delete":
      return null;
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

    // ---- misc ----
    case "set_tray_language":
    case "open_data_dir":
    case "open_models_dir":
    case "open_external":
      return null;
    case "cancel_generation":
    case "cancel_download":
      return null;
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
