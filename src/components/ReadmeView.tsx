// A model's README in the store, read the way Hugging Face shows it: its
// Markdown and the HTML inside it — badge rows, centred banners, <ul> lists,
// tables, <details> — rendered rather than stripped to loose lines. The HTML
// is the repo author's, so it is sanitised to GitHub's allow-list first
// (no scripts, event handlers, styles or frames), and relative links and
// images point back into the repo.
import { useCallback, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import rehypeHighlight from "rehype-highlight";
import { hfBase } from "../lib/ipc";

const remarkPlugins = [remarkGfm];
/** GitHub's allow-list, and elements whose contents go with them: by default
 *  only a <script>'s do, and a <style> left its CSS on the page as text. */
const schema = {
  ...defaultSchema,
  strip: [...new Set([...(defaultSchema.strip ?? []), "script", "style", "noscript", "template", "title", "iframe", "object", "embed", "svg", "math", "textarea", "select"])],
};
type HNode = { type: string; tagName?: string; value?: string; properties?: Record<string, unknown>; children?: HNode[] };
const ALERT = /^\s*\[!(note|tip|important|warning|caution)\][ \t]*\n?/i;

/** GitHub's alerts — a quote that opens with [!NOTE], [!TIP], [!IMPORTANT],
 *  [!WARNING] or [!CAUTION] — as the callouts Hugging Face draws, rather than
 *  a quote that starts with the marker in brackets. */
function rehypeAlerts() {
  const visit = (node: HNode) => {
    for (const child of node.children ?? []) {
      if (child.type !== "element") continue;
      if (child.tagName === "blockquote") alert(child);
      visit(child);
    }
  };
  const alert = (quote: HNode) => {
    const kids = quote.children ?? [];
    const first = kids.find((k) => k.type === "element");
    const text = first?.tagName === "p" ? first.children?.[0] : undefined;
    const m = text?.type === "text" ? text.value?.match(ALERT) : null;
    if (!first || !text || !m) return;
    text.value = text.value!.slice(m[0].length);
    // The marker on a line of its own leaves an empty paragraph behind.
    if (first.children!.length === 1 && !text.value.trim()) kids.splice(kids.indexOf(first), 1);
    const kind = m[1].toLowerCase();
    quote.properties = { ...quote.properties, className: ["readme-alert", `readme-alert-${kind}`] };
    const title = kind[0].toUpperCase() + kind.slice(1);
    kids.unshift({ type: "element", tagName: "p", properties: { className: ["readme-alert-title"] }, children: [{ type: "text", value: title }] });
    quote.children = kids;
  };
  return (tree: HNode) => visit(tree);
}

// Sanitise before the app's own touches: the alert classes and the
// highlighter's spans are ours.
const rehypePlugins = [rehypeRaw, [rehypeSanitize, schema], rehypeAlerts, [rehypeHighlight, { ignoreMissing: true }]] as const;

/** Where a README's link or image goes: as written when it is absolute, into
 *  the repo when it is relative (a file's page for a link, its bytes for an
 *  image), nowhere when it is any other scheme. */
export function readmeUrl(url: string, attr: string, repo: string): string {
  if (/^(https?:|mailto:)/i.test(url) || url.startsWith("#")) return defaultUrlTransform(url);
  if (url.startsWith("//")) return defaultUrlTransform(`https:${url}`);
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return "";
  const path = url.replace(/^\.?\/+/, "");
  return `${hfBase()}/${repo}/${attr === "src" ? "resolve" : "blob"}/main/${path}`;
}

/** Banners and charts are big and below the fold: fetched as they come into
 *  view. A broken one is hidden by the store pane (its error capture). */
const components = {
  // An image whose address the sanitiser or the URL rule took away is not
  // drawn at all, rather than as an empty box.
  img: ({ node: _node, ...props }: ComponentPropsWithoutRef<"img"> & { node?: unknown }) =>
    props.src ? <img {...props} alt={props.alt ?? ""} loading="lazy" decoding="async" /> : null,
};

export function ReadmeView({ text, repo }: { text: string; repo: string }) {
  const urlTransform = useCallback((url: string, key: string) => readmeUrl(url, key, repo), [repo]);
  return (
    <ReactMarkdown
      remarkPlugins={remarkPlugins}
      rehypePlugins={rehypePlugins as never}
      urlTransform={urlTransform}
      components={components}
    >
      {text}
    </ReactMarkdown>
  );
}
