// Cross-session memory (2.0 M4): facts as files under
// <workspace>/.chaty/memory/, following the same economics as skills — the
// system prompt carries a capped INDEX (one line per fact), bodies are plain
// workspace files the model reads with read_file only when a line looks
// relevant. Plain markdown, human-editable, never leaves the machine.
//
// Layout:
//   .chaty/memory/MEMORY.md        — the index: "- [title](file.md) — hook"
//   .chaty/memory/<slug>.md        — one fact per file
//
// Writing goes through the `remember` tool (registered only when the
// workspace has memory enabled — a session without it keeps a byte-identical
// prompt, enforced by the golden test, same trick as skills).

export const MEMORY_DIR = ".chaty/memory";
export const MEMORY_INDEX = `${MEMORY_DIR}/MEMORY.md`;
/** Index budget in chars — the prompt tax must stay flat no matter how much
 *  a project remembers. ~30 lines; older lines beyond the cap are dropped
 *  from the PROMPT (never from disk). */
const INDEX_CAP = 2200;

export function slugify(title: string): string {
  const s = title
    .toLowerCase()
    .replace(/[^a-z0-9一-鿿]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return s || "note";
}

/** The prompt block. Empty index ⇒ "" (prompt stays byte-identical). */
export function memoryIndexDoc(index: string, lang: "zh" | "en"): string {
  const body = index.trim();
  if (!body) return "";
  const head =
    lang === "zh"
      ? `\n\n项目记忆(往届会话留下的事实索引;条目相关时用 read_file 读正文,不要凭索引行猜细节):`
      : `\n\nProject memory (facts left by earlier sessions; read_file an entry when it's relevant — don't guess details from the index line):`;
  const capped = body.length <= INDEX_CAP ? body : body.slice(0, INDEX_CAP);
  // Never cut mid-line: drop the trailing partial line if capping hit one.
  const clean = body.length <= INDEX_CAP ? capped : capped.slice(0, capped.lastIndexOf("\n") + 1);
  return `${head}\n${clean.trimEnd()}`;
}

/** Nudge appended (only when memory is on) so small models actually write. */
export function memoryWriteNudge(lang: "zh" | "en"): string {
  return lang === "zh"
    ? `\n- 收尾前,把下次会话会用到的**非显而易见**发现用 remember 存档(坑、约定、决定);任务本身的代码改动不用存。`
    : `\n- Before wrapping up, remember() the NON-OBVIOUS findings a future session will need (pitfalls, conventions, decisions) — not the code changes themselves.`;
}

export interface MemoryFs {
  /** Read a workspace-relative file; reject if missing. */
  readFile: (path: string) => Promise<string>;
  /** Create/overwrite a workspace-relative file (mkdir -p semantics). */
  writeFile: (path: string, content: string) => Promise<void>;
}

/** Load the index for prompt injection. Missing file ⇒ "" (feature dormant). */
export async function loadMemoryIndex(fs: Pick<MemoryFs, "readFile">): Promise<string> {
  try {
    // Strip hashline anchors — same guard as project docs and skills.
    return (await fs.readFile(MEMORY_INDEX)).replace(/^\d+:[a-z]{2,4}→/gm, "");
  } catch {
    return "";
  }
}

/** Persist one fact: write the file, then upsert its index line. Returns the
 *  model-facing confirmation. Titles are upserted, not duplicated — calling
 *  remember twice with the same title updates the fact in place. */
export async function rememberFact(
  fs: MemoryFs,
  title: string,
  fact: string,
  lang: "zh" | "en",
): Promise<string> {
  const t = title.trim().slice(0, 80);
  const body = fact.trim();
  if (!t || !body) {
    return lang === "zh"
      ? 'ERROR: 需要 "title" 和 "fact" 两个参数,例如 {"title":"构建规矩","fact":"发版前必须跑 scripts/gate.sh"}'
      : 'ERROR: both "title" and "fact" are required, e.g. {"title":"build rule","fact":"run scripts/gate.sh before any release"}';
  }
  const file = `${slugify(t)}.md`;
  const hook = body.replace(/\s+/g, " ").slice(0, 90);
  await fs.writeFile(`${MEMORY_DIR}/${file}`, `# ${t}\n\n${body.slice(0, 4000)}\n`);

  let index = "";
  try {
    index = await fs.readFile(MEMORY_INDEX);
  } catch {
    /* first fact — fresh index */
  }
  const line = `- [${t}](${file}) — ${hook}`;
  const kept = index
    .split("\n")
    .filter((l) => l.trim() && !l.includes(`](${file})`));
  // Newest first: the cap in memoryIndexDoc drops the OLDEST lines.
  const next = [line, ...kept].join("\n") + "\n";
  await fs.writeFile(MEMORY_INDEX, next);
  return lang === "zh"
    ? `已记住:${t}(${MEMORY_DIR}/${file};索引已更新)`
    : `Remembered: ${t} (${MEMORY_DIR}/${file}; index updated)`;
}
