/**
 * Shared bench plumbing: the chaty-headless stdio bridge and the answer
 * normalizer, both of which existed as near-copies in every runner.
 *
 * The copies weren't harmless. The MCP runner's Bridge drifted on the
 * callback contract and blew up on `onPrefill`; graders and the MCP runner
 * each kept their own `norm()`, and a grader that normalizes differently
 * from its sibling is how a correct answer gets marked wrong (this suite has
 * already had two such miscarriages).
 */
import { spawn, type ChildProcess } from "node:child_process";

export type Json = Record<string, unknown>;

/** Line-delimited JSON over chaty-headless's stdin/stdout. */
export class Bridge {
  private proc: ChildProcess;
  private buf = "";
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void; onEvent?: (ev: unknown) => void }
  >();
  private nextId = 1;

  constructor(bin: string, env?: Record<string, string>) {
    this.proc = spawn(bin, [], {
      stdio: ["pipe", "pipe", "inherit"],
      // Benches must NEVER pop a visible Chrome in the user's face — the
      // original web runner set this and the extraction dropped it, which
      // put a browser window on the owner's screen every oracle task until
      // they complained. Default here so every runner inherits it; a caller
      // can still override explicitly.
      env: { ...process.env, CHATY_BROWSER_HEADLESS: "1", ...env },
    });
    this.proc.stdout!.on("data", (chunk: Buffer) => {
      this.buf += chunk.toString();
      let nl: number;
      while ((nl = this.buf.indexOf("\n")) >= 0) {
        const line = this.buf.slice(0, nl);
        this.buf = this.buf.slice(nl + 1);
        if (!line.trim()) continue;
        let msg: Json;
        try {
          msg = JSON.parse(line);
        } catch {
          continue;
        }
        const p = this.pending.get(msg.id as number);
        if (!p) continue;
        if (msg.event) {
          p.onEvent?.(msg.event);
          continue;
        }
        this.pending.delete(msg.id as number);
        if (msg.error) p.reject(new Error(String(msg.error)));
        else p.resolve(msg.result);
      }
    });
  }

  call(cmd: string, args: Json, onEvent?: (ev: unknown) => void): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
      this.proc.stdin!.write(JSON.stringify({ id, cmd, args }) + "\n");
    });
  }

  /** Route Tauri IPC to the bridge — identical in every runner, including the
   *  `generate` streaming special case that is easy to get subtly wrong. */
  async ipc(cmd: string, args?: Json): Promise<unknown> {
    if (cmd === "generate") {
      const ch = (args as Json)?.onEvent as { onmessage?: (ev: unknown) => void } | undefined;
      return this.call("generate", { request: (args as Json)?.request }, (ev) => ch?.onmessage?.(ev));
    }
    return this.call(cmd, args ?? {});
  }

  kill(): void {
    this.proc.kill();
  }
}

/** Answer normalization for substring grading: fold case and strip markdown
 *  emphasis — a model writing "maximum of **5** days" must not fail a "5"
 *  check (that exact false negative cost a task a wrongful ✗ once).
 *
 *  Deliberately byte-identical to the semantics graders.mts already shipped —
 *  no whitespace collapsing, no trimming. A grading instrument must not
 *  change judgement as a side effect of deduplication; if it should become
 *  more permissive, that is its own change, with its own re-grade. */
export function norm(s: string): string {
  return s.toLowerCase().replace(/[*_`]/g, "");
}
