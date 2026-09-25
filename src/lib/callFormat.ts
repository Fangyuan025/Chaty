/** The tool-call format the current agent turn teaches — the one the loaded
 *  model's chat template was trained on (ModelInfo.toolFormat), or the
 *  fallback the user picked for a family whose template names none:
 *
 *    xml    Qwen3.5/3.6/3.8   <tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>\n</tool_call>
 *    json   Qwen3, QwQ        <tool_call>{"name":"…","arguments":{…}}</tool_call>
 *    gemma  Gemma 4           <|tool_call>call:name{key:<|"|>text<|"|>,n:5}<tool_call|>
 *    lfm    LFM2              <|tool_call_start|>[name(key="value")]<|tool_call_end|>
 *    minicpm MiniCPM5         <function name="name"><param name="key">value</param></function>
 *    ifm    K2 Horizon        <ifm|tool_calls>\n<ifm|tool_call>name\n<ifm|arg_key>key</ifm|arg_key>\n<ifm|arg_value>value</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>
 *    glm    GLM-4.5/4.6/4.7   <tool_call>name\n<arg_key>key</arg_key>\n<arg_value>value</arg_value>\n</tool_call>
 *
 *  Every format is always PARSED; this decides what the prompt and every
 *  correction teach, so none of them teaches a model a format it was not
 *  trained on. Set at the start of each turn. */
export type CallFormat = "json" | "xml" | "gemma" | "lfm" | "ifm" | "glm" | "minicpm";

let format: CallFormat = "json";

/** The format a turn uses: the model's own when the setting is "auto" and its
 *  template names one, else the fallback picked for unknown families; a format
 *  chosen by hand overrides both. */
export function resolveToolFormat(
  setting: "auto" | CallFormat,
  native: CallFormat | null | undefined,
  fallback: "xml" | "json",
): CallFormat {
  return setting === "auto" ? (native ?? fallback) : setting;
}

export function setCallFormat(f: CallFormat): void {
  format = f;
}

export function callFormat(): CallFormat {
  return format;
}

/** How the system prompt tells the model to write a call. */
export function callRule(zh: boolean, f: CallFormat): string {
  switch (f) {
    case "xml":
      return zh
        ? "- 每次只调用一个工具。想清楚要做什么之后,按下面的格式输出这一个调用就立即停止——同一条消息里不要再写说明文字,也不要写第二个调用(思考不算):\n<tool_call>\n<function=工具名>\n<parameter=参数名>\n参数值(可以多行,原样书写,不需要任何转义)\n</parameter>\n</function>\n</tool_call>\n  必填参数一个都不能少,每个参数一对 <parameter=…></parameter>;数组或对象类型的参数(比如 multi_edit 的 edits)写成 JSON。"
        : "- Call ONE tool at a time. Think it through, then write the one call in this form and STOP immediately — no prose and no second call in that message (your reasoning does not count):\n<tool_call>\n<function=tool_name>\n<parameter=argument_name>\nthe value (any number of lines, written as is, nothing escaped)\n</parameter>\n</function>\n</tool_call>\n  Every required argument must be there, each in its own <parameter=…></parameter>; an array or object argument (multi_edit's edits, say) is written as JSON.";
    case "gemma":
      return zh
        ? '- 每次只调用一个工具。想清楚要做什么之后,按下面的格式输出这一个调用就立即停止——同一条消息里不要再写说明文字,也不要写第二个调用(思考不算):\n<|tool_call>call:工具名{参数名:<|"|>文字值<|"|>,数字参数:5,开关参数:true}<tool_call|>\n  文字值放在两个 <|"|> 之间,原样书写(可以多行,不需要任何转义);必填参数一个都不能少;数组写成 [..],对象写成 {..}。'
        : '- Call ONE tool at a time. Think it through, then write the one call in this form and STOP immediately — no prose and no second call in that message (your reasoning does not count):\n<|tool_call>call:tool_name{argument_name:<|"|>text value<|"|>,number_argument:5,flag_argument:true}<tool_call|>\n  A text value goes between two <|"|>, written as is (any number of lines, nothing escaped); every required argument must be there; arrays are [..] and objects {..}.';
    case "lfm":
      return zh
        ? '- 每次只调用一个工具。想清楚要做什么之后,按下面的格式输出这一个调用就立即停止——同一条消息里不要再写说明文字,也不要写第二个调用(思考不算):\n<|tool_call_start|>[工具名(参数名="文字值", 数字参数=5)]<|tool_call_end|>\n  必填参数一个都不能少;文字值里的双引号写成 \\",换行写成 \\n。'
        : '- Call ONE tool at a time. Think it through, then write the one call in this form and STOP immediately — no prose and no second call in that message (your reasoning does not count):\n<|tool_call_start|>[tool_name(argument_name="text value", number_argument=5)]<|tool_call_end|>\n  Every required argument must be there; inside a text value write a double quote as \\" and a newline as \\n.';
    case "minicpm":
      return zh
        ? '- 每次只调用一个工具。想清楚要做什么之后,按下面的格式输出这一个调用就立即停止——同一条消息里不要再写说明文字,也不要写第二个调用(思考不算):\n<function name="工具名"><param name="参数名">参数值</param></function>\n  必填参数一个都不能少,每个参数一对 <param name="…"></param>;参数值里有 <、& 或换行时,用 <![CDATA[ … ]]> 包住;数组或对象类型的参数(比如 multi_edit 的 edits)写成 JSON。'
        : '- Call ONE tool at a time. Think it through, then write the one call in this form and STOP immediately — no prose and no second call in that message (your reasoning does not count):\n<function name="tool_name"><param name="argument_name">the value</param></function>\n  Every required argument must be there, each in its own <param name="…"></param>; a value holding <, & or a line break goes inside <![CDATA[ … ]]>; an array or object argument (multi_edit\'s edits, say) is written as JSON.';
    case "ifm":
      return zh
        ? "- 每次只调用一个工具。想清楚要做什么之后,按下面的格式输出这一个调用就立即停止——同一条消息里不要再写说明文字,也不要写第二个调用(思考不算):\n<ifm|tool_calls>\n<ifm|tool_call>工具名\n<ifm|arg_key>参数名</ifm|arg_key>\n<ifm|arg_value>参数值(可以多行,原样书写,不需要任何转义)</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>\n  必填参数一个都不能少,每个参数一对 arg_key/arg_value;数组或对象类型的参数(比如 multi_edit 的 edits)写成 JSON。"
        : "- Call ONE tool at a time. Think it through, then write the one call in this form and STOP immediately — no prose and no second call in that message (your reasoning does not count):\n<ifm|tool_calls>\n<ifm|tool_call>tool_name\n<ifm|arg_key>argument_name</ifm|arg_key>\n<ifm|arg_value>the value (any number of lines, written as is, nothing escaped)</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>\n  Every required argument must be there, each as an arg_key/arg_value pair; an array or object argument (multi_edit's edits, say) is written as JSON.";
    case "glm":
      return zh
        ? "- 每次只调用一个工具。想清楚要做什么之后,按下面的格式输出这一个调用就立即停止——同一条消息里不要再写说明文字,也不要写第二个调用(思考不算):\n<tool_call>工具名\n<arg_key>参数名</arg_key>\n<arg_value>参数值(可以多行,原样书写,不需要任何转义)</arg_value>\n</tool_call>\n  必填参数一个都不能少,每个参数一对 arg_key/arg_value;数组或对象类型的参数(比如 multi_edit 的 edits)写成 JSON。"
        : "- Call ONE tool at a time. Think it through, then write the one call in this form and STOP immediately — no prose and no second call in that message (your reasoning does not count):\n<tool_call>tool_name\n<arg_key>argument_name</arg_key>\n<arg_value>the value (any number of lines, written as is, nothing escaped)</arg_value>\n</tool_call>\n  Every required argument must be there, each as an arg_key/arg_value pair; an array or object argument (multi_edit's edits, say) is written as JSON.";
    default:
      return zh
        ? '- 每次只调用一个工具。想清楚要做什么之后,只输出一行 <tool_call>{"name":"工具名","arguments":{...}}</tool_call> 就立即停止——同一条消息里不要再写说明文字,也不要写第二个调用(思考不算)。'
        : '- Call ONE tool at a time. Think it through, then output a single line <tool_call>{"name":"tool","arguments":{...}}</tool_call> and STOP immediately — no prose and no second call in that message (your reasoning does not count).';
  }
}

/** The closing marker of each format — generation stops at whichever comes. */
export const CALL_CLOSERS = ["</tool_call>", "<tool_call|>", "<|tool_call_end|>", "</ifm|tool_call>"];

/** Where generation stops for a turn taught `f`: every format's closer, plus
 *  MiniCPM5's `</function>` — which only closes a call in that format; in the
 *  XML one it is followed by `</tool_call>`, which is where that call ends. */
export function callClosers(f: CallFormat): string[] {
  return f === "minicpm" ? [...CALL_CLOSERS, "</function>"] : [...CALL_CLOSERS];
}

// Inner before outer: K2's call closes, then the block around it — with the
// newline its template writes before the block's closer.
const CALL_PAIRS: [string, string][] = [
  ["<tool_call>", "</tool_call>"],
  ["<|tool_call>", "<tool_call|>"],
  ["<|tool_call_start|>", "<|tool_call_end|>"],
  ["<ifm|tool_call>", "</ifm|tool_call>"],
  ["<ifm|tool_calls>", "\n</ifm|tool_calls>"],
  ["<function name=", "</function>"],
];

/** Where the first tool call in a model's output begins, in any format; -1
 *  when there is none. */
export function callStart(text: string): number {
  let at = -1;
  for (const m of ["<tool_call>", "<|tool_call>", "<|tool_call_start|>", "<function=", "<ifm|tool_call", "<function name="]) {
    const i = text.indexOf(m);
    if (i !== -1 && (at === -1 || i < at)) at = i;
  }
  return at;
}

/** `text` without the tool calls written in it — reasoning as a person reads
 *  it, where a call is markup, not thought (its step card shows it). A call
 *  still being written runs to the end. */
export function withoutCalls(text: string): string {
  let out = "";
  let rest = text;
  for (;;) {
    const c = callStart(rest);
    if (c === -1) return out + rest;
    out += rest.slice(0, c);
    const tail = rest.slice(c);
    const close = CALL_PAIRS.find(([o]) => tail.startsWith(o))?.[1] ?? "</function>";
    const k = tail.indexOf(close);
    if (k === -1) return out;
    rest = tail.slice(k + close.length);
  }
}

/** Put back the closer the stop sequence trimmed off, in the format the call
 *  was opened in — the model wrote it and the engine's cache holds it. */
export function closeOpenCalls(turn: string): string {
  for (const [open, close] of CALL_PAIRS) {
    if (turn.split(open).length > turn.split(close).length) turn += close;
  }
  return turn;
}

/** The format a call in a model's output was written in. */
export function formatOf(text: string): CallFormat {
  if (text.includes("<ifm|tool_call")) return "ifm";
  if (/<tool_call>\s*[A-Za-z_][\w.-]*\s*(?:<arg_key>|<\/tool_call>)/.test(text)) return "glm";
  if (/<function\s+name\s*=/.test(text)) return "minicpm";
  if (text.includes("<|tool_call>")) return "gemma";
  if (text.includes("<|tool_call_start|>")) return "lfm";
  const at = text.indexOf("<tool_call>");
  const body = at === -1 ? text : text.slice(at + "<tool_call>".length);
  return !/^\s*\{/.test(body) && body.includes("<function=") ? "xml" : "json";
}

/** A whole call as the model writes it in this format — for the calls Chaty
 *  itself puts into history. A turn stored with its call re-written as JSON is
 *  a JSON call in the model's own hand, the strongest example there is: Gemma 4
 *  switched to JSON from its second call on because each of its native calls
 *  came back in its history followed by a JSON copy. */
export function renderCall(name: string, args: Record<string, unknown>, f: CallFormat = format): string {
  const entries = Object.entries(args);
  switch (f) {
    case "xml":
      return `<tool_call>\n<function=${name}>\n${entries
        .map(([k, v]) => `<parameter=${k}>\n${typeof v === "string" ? v : JSON.stringify(v)}\n</parameter>\n`)
        .join("")}</function>\n</tool_call>`;
    case "gemma":
      return `<|tool_call>call:${name}{${entries.map(([k, v]) => `${k}:${gemmaValue(v)}`).join(",")}}<tool_call|>`;
    case "lfm":
      return `<|tool_call_start|>[${name}(${entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})]<|tool_call_end|>`;
    case "minicpm":
      return `<function name="${name}">${entries.map(([k, v]) => `<param name="${k}">${minicpmValue(v)}</param>`).join("")}</function>`;
    case "glm":
      return `<tool_call>${name}\n${entries
        .map(([k, v]) => `<arg_key>${k}</arg_key>\n<arg_value>${typeof v === "string" ? v : JSON.stringify(v)}</arg_value>\n`)
        .join("")}</tool_call>`;
    case "ifm":
      return `<ifm|tool_calls>\n<ifm|tool_call>${name}\n${entries
        .map(([k, v]) => `<ifm|arg_key>${k}</ifm|arg_key>\n<ifm|arg_value>${typeof v === "string" ? v : JSON.stringify(v)}</ifm|arg_value>\n`)
        .join("")}</ifm|tool_call>\n</ifm|tool_calls>`;
    default:
      return `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;
  }
}

/** "a <tool_call>" as a nudge names it: the JSON wording as it always was,
 *  the format's own call for the others. */
export function callTag(zh: boolean): string {
  if (format === "json") return zh ? "一行 <tool_call>" : "a single <tool_call> line";
  return oneCall(zh);
}

/** "a tool call", spelled the way this turn's format writes one. */
export function oneCall(zh: boolean): string {
  switch (format) {
    case "xml":
      return zh ? "一个 <tool_call><function=…> 工具调用" : "one <tool_call><function=…> tool call";
    case "gemma":
      return zh ? "一个 <|tool_call>call:工具名{…}<tool_call|> 工具调用" : "one <|tool_call>call:tool_name{…}<tool_call|> tool call";
    case "lfm":
      return zh ? "一个 <|tool_call_start|>[工具名(…)]<|tool_call_end|> 工具调用" : "one <|tool_call_start|>[tool_name(…)]<|tool_call_end|> tool call";
    case "ifm":
      return zh ? "一个 <ifm|tool_call>工具名 …</ifm|tool_call> 工具调用" : "one <ifm|tool_call>tool_name …</ifm|tool_call> tool call";
    case "glm":
      return zh ? "一个 <tool_call>工具名<arg_key>…</tool_call> 工具调用" : "one <tool_call>tool_name<arg_key>…</tool_call> tool call";
    case "minicpm":
      return zh ? '一个 <function name="工具名">…</function> 工具调用' : 'one <function name="tool_name">…</function> tool call';
    default:
      return zh
        ? '一行 <tool_call>{"name":"...","arguments":{...}}</tool_call>'
        : 'a single line <tool_call>{"name":"...","arguments":{...}}</tool_call>';
  }
}

/** A value the way MiniCPM5's template writes it: as is, unless it holds `<`,
 *  `&` or a line break, which go inside CDATA; anything not text as JSON. */
function minicpmValue(v: unknown): string {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return typeof v === "string" && /[<&\n]/.test(s) ? `<![CDATA[${s}]]>` : s;
}

/** A value the way Gemma 4's template writes it: text between <|"|>, numbers
 *  and booleans bare, arrays and objects in brackets, keys unquoted. */
function gemmaValue(v: unknown): string {
  if (typeof v === "string") return `<|"|>${v}<|"|>`;
  if (Array.isArray(v)) return `[${v.map(gemmaValue).join(",")}]`;
  if (v && typeof v === "object") {
    return `{${Object.entries(v as Record<string, unknown>).map(([k, x]) => `${k}:${gemmaValue(x)}`).join(",")}}`;
  }
  return String(v);
}

/** One call, written the way this turn's format writes it, on one line — for
 *  the examples inside corrections and hints. A JSON example in a turn that
 *  teaches another format is what pulled Gemma 4 back to JSON after its
 *  first native call. */
export function callExample(name: string, json: string): string {
  let o: Record<string, unknown>;
  try {
    o = JSON.parse(json) as Record<string, unknown>;
  } catch {
    return `${name} ${json}`;
  }
  const entries = Object.entries(o);
  switch (format) {
    case "xml":
      return `<function=${name}>${entries
        .map(([k, v]) => `<parameter=${k}>${typeof v === "string" ? v : JSON.stringify(v)}</parameter>`)
        .join("")}</function>`;
    case "gemma":
      return `call:${name}{${entries.map(([k, v]) => `${k}:${gemmaValue(v)}`).join(",")}}`;
    case "lfm":
      return `${name}(${entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})`;
    case "minicpm":
      return `<function name="${name}">${entries.map(([k, v]) => `<param name="${k}">${minicpmValue(v)}</param>`).join("")}</function>`;
    case "glm":
      return `<tool_call>${name}${entries
        .map(([k, v]) => `<arg_key>${k}</arg_key><arg_value>${typeof v === "string" ? v : JSON.stringify(v)}</arg_value>`)
        .join("")}</tool_call>`;
    case "ifm":
      return `<ifm|tool_call>${name} ${entries
        .map(([k, v]) => `<ifm|arg_key>${k}</ifm|arg_key><ifm|arg_value>${typeof v === "string" ? v : JSON.stringify(v)}</ifm|arg_value>`)
        .join("")}</ifm|tool_call>`;
    default:
      return `${name} ${json}`;
  }
}

/** A tool doc line with its argument list written plainly — `args: { "path":
 *  string, "offset"?: number }` becomes `args: path: string, offset?: number`
 *  — for a turn that teaches a format other than JSON, where a JSON object in
 *  every tool's description is a JSON example on every line of the prompt. */
export function plainArgs(line: string): string {
  const at = line.indexOf("args: {");
  const end = line.lastIndexOf("}");
  if (at === -1 || end < at) return line;
  const inner = line
    .slice(at + "args: {".length, end)
    .trim()
    .replace(/"([A-Za-z_][A-Za-z0-9_]*)"(\??):/g, "$1$2:")
    // Names listed without a type (`"path", "old_string"`) too: a quoted name
    // in the docs came back as `<parameter="path">`.
    .replace(/"([A-Za-z_][A-Za-z0-9_]*)"/g, "$1");
  return `${line.slice(0, at)}args: ${inner}${line.slice(end + 1)}`;
}

/** A JSON argument example ({"path":"src/app.ts"}) in this turn's format. */
export function argsExample(json: string): string {
  if (format === "json") return `arguments: ${json}`;
  let o: Record<string, unknown>;
  try {
    o = JSON.parse(json) as Record<string, unknown>;
  } catch {
    return `arguments: ${json}`;
  }
  const entries = Object.entries(o);
  switch (format) {
    case "xml":
      return entries.map(([k, v]) => `<parameter=${k}>\n${typeof v === "string" ? v : JSON.stringify(v)}\n</parameter>`).join("\n");
    case "gemma":
      return entries.map(([k, v]) => `${k}:${gemmaValue(v)}`).join(",");
    case "lfm":
      return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
    case "ifm":
      return entries
        .map(([k, v]) => `<ifm|arg_key>${k}</ifm|arg_key>\n<ifm|arg_value>${typeof v === "string" ? v : JSON.stringify(v)}</ifm|arg_value>`)
        .join("\n");
    case "glm":
      return entries.map(([k, v]) => `<arg_key>${k}</arg_key>\n<arg_value>${typeof v === "string" ? v : JSON.stringify(v)}</arg_value>`).join("\n");
    case "minicpm":
      return entries.map(([k, v]) => `<param name="${k}">${minicpmValue(v)}</param>`).join("");
  }
  return `arguments: ${json}`;
}
