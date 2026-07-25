// MCP bridge (2.0 M1): server configs live in localStorage like every other
// setting; the Rust side owns processes and wire protocol. This module turns
// a connected server's tools into ToolSpecs the agent can call — with docs
// SYNTHESIZED LEAN: community MCP schemas run thousands of tokens, which is
// exactly how 19K-token system prompts stop fitting small models. Every MCP
// tool registers untrusted (results are external content) and, unless the
// user marks the server trusted, mutating (each call rides the approval UI).

import { invoke } from "@tauri-apps/api/core";
import { registerTool, unregisterTool, type ToolSpec } from "./toolRegistry";

export interface McpTransportCfg {
  transport: "stdio" | "http";
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
}

export interface McpServerCfg extends McpTransportCfg {
  /** Short alias; prefixes tool names (`gh__search_issues`). */
  name: string;
  enabled: boolean;
  /** Trusted servers skip per-call approval (user opt-in, off by default). */
  trusted?: boolean;
}

export interface McpToolInfo {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

const LS_KEY = "chaty.mcpServers";
/** A server with more tools than this goes deferred: one index line in the
 *  prompt instead of N doc lines — the context budget stays fixed no matter
 *  how many tools a server brings. */
const CORE_TIER_MAX_TOOLS = 8;

export function loadMcpServers(): McpServerCfg[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    const arr = raw ? (JSON.parse(raw) as McpServerCfg[]) : [];
    return Array.isArray(arr) ? arr.filter((s) => s && typeof s.name === "string") : [];
  } catch {
    return [];
  }
}

export function saveMcpServers(servers: McpServerCfg[]): void {
  localStorage.setItem(LS_KEY, JSON.stringify(servers));
}

// ── Lean-doc synthesis ───────────────────────────────────────────────────────

/** First sentence of a description, hard-capped — community descriptions run
 *  to paragraphs and every char here is a per-step prefill tax. */
export function leanDescription(desc: string, cap = 110): string {
  const flat = desc.replace(/\s+/g, " ").trim();
  const stop = flat.search(/[.。!?]\s|[.。!?]$/);
  const first = stop === -1 ? flat : flat.slice(0, stop + 1);
  return first.length <= cap ? first : `${first.slice(0, cap - 1)}…`;
}

interface JsonSchemaish {
  type?: string;
  properties?: Record<string, { type?: string | string[]; description?: string }>;
  required?: string[];
}

/** Compact args signature from a JSON schema: required props bare, optionals
 *  marked `?`. `{ "query": string, "page"?: number }` — the shape the native
 *  tool docs already teach the model to read. */
export function leanArgs(schema: Record<string, unknown>): string {
  const s = schema as JsonSchemaish;
  const props = s.properties ?? {};
  const required = new Set(s.required ?? []);
  const parts = Object.entries(props).map(([key, p]) => {
    const t = Array.isArray(p.type) ? p.type[0] : (p.type ?? "any");
    return `"${key}"${required.has(key) ? "" : "?"}: ${t}`;
  });
  return parts.length ? `{ ${parts.join(", ")} }` : "{}";
}

export function leanDocLine(toolName: string, info: McpToolInfo): { zh: string; en: string } {
  const desc = leanDescription(info.description || info.name);
  const line = `- ${toolName}: ${desc} args: ${leanArgs(info.input_schema)}`;
  return { zh: line, en: line }; // MCP servers are monolingual; same line both ways
}

/** Full doc for the deferred-tier load path: complete description + schema.
 *  Returned as the correction when the model calls an indexed-but-unloaded
 *  tool wrong — the same missing-arg recovery pattern natives already use. */
export function fullDoc(toolName: string, info: McpToolInfo): string {
  return (
    `${toolName}: ${info.description.trim() || "(no description)"}\n` +
    `args schema: ${JSON.stringify(info.input_schema)}`
  );
}

// ── Registration & dispatch ──────────────────────────────────────────────────

/** Live routing table: model-facing name → server/tool/schema. */
const active = new Map<string, { server: string; tool: string; info: McpToolInfo }>();

export function mcpToolName(server: string, tool: string): string {
  const safe = (s: string) => s.replace(/[^a-zA-Z0-9_]/g, "_");
  return `${safe(server)}__${safe(tool)}`;
}

export function activeMcpTools(): string[] {
  return [...active.keys()];
}

function unregisterServerTools(server: string): void {
  for (const [name, entry] of [...active]) {
    if (entry.server === server) {
      unregisterTool(name);
      active.delete(name);
    }
  }
}

/** Connect every enabled server and (re)register its tools. Returns per-server
 *  status for the settings UI. Disabled/removed servers unregister. */
export async function syncMcpServers(
  servers: McpServerCfg[] = loadMcpServers(),
): Promise<{ server: string; tools: number; error?: string }[]> {
  const out: { server: string; tools: number; error?: string }[] = [];
  const wanted = new Set(servers.filter((s) => s.enabled).map((s) => s.name));
  for (const server of new Set([...active.values()].map((e) => e.server))) {
    if (!wanted.has(server)) {
      unregisterServerTools(server);
      void invoke("mcp_disconnect", { name: server }).catch(() => {});
    }
  }
  for (const cfg of servers) {
    if (!cfg.enabled) continue;
    try {
      const transport =
        cfg.transport === "http"
          ? { transport: "http", url: cfg.url ?? "", headers: cfg.headers ?? {} }
          : {
              transport: "stdio",
              command: cfg.command ?? "",
              args: cfg.args ?? [],
              env: cfg.env ?? {},
            };
      const tools = await invoke<McpToolInfo[]>("mcp_connect", {
        name: cfg.name,
        transport,
      });
      unregisterServerTools(cfg.name); // reconnect = clean slate
      const tier = tools.length <= CORE_TIER_MAX_TOOLS ? "core" : "deferred";
      for (const info of tools) {
        const name = mcpToolName(cfg.name, info.name);
        const required = (info.input_schema as JsonSchemaish).required ?? [];
        const spec: ToolSpec = {
          name,
          source: "mcp",
          suite: "core",
          perm: "network",
          tier,
          docLine: leanDocLine(name, info),
          untrusted: true,
          mutating: cfg.trusted !== true,
          requiredArgs: required.length ? required : undefined,
          hint: { zh: leanDescription(info.description, 40), en: leanDescription(info.description, 40) },
        };
        if (registerTool(spec)) active.set(name, { server: cfg.name, tool: info.name, info });
      }
      out.push({ server: cfg.name, tools: tools.length });
    } catch (e) {
      out.push({ server: cfg.name, tools: 0, error: String(e) });
    }
  }
  return out;
}

/** Dispatch one MCP tool call (agentLoop's default branch routes here).
 *  Validates required args from the schema first: for a DEFERRED tool this
 *  doubles as the load path — the correction carries the full doc, and the
 *  model's retry pattern (trained by the native missing-arg guard) does the
 *  rest. */
export async function callMcpTool(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  const entry = active.get(name);
  if (!entry) return `ERROR: MCP tool ${name} is not connected (its server may be disabled).`;
  const missing = (entry.info.input_schema as JsonSchemaish).required?.filter(
    (k) => args[k] === undefined || args[k] === "",
  );
  if (missing?.length) {
    return `ERROR: missing "${missing[0]}". Tool reference:\n${fullDoc(name, entry.info)}`;
  }
  return await invoke<string>("mcp_call", {
    server: entry.server,
    tool: entry.tool,
    args,
  });
}
