/** The last user message as the model receives it: this turn's context (the
 *  date line, retrieved passages) first, then the question. When there are
 *  passages the question is set apart and named. Issue #15: after a block of
 *  retrieved text, a bare "官方站" read as one more line of the document — the
 *  model summarised the passages or answered the previous question instead
 *  (5 of 6 samples on Qwen3.5-4B; 0 of 6 once the question was labelled). */
export function turnMessage(parts: string[], question: string, label?: string): string {
  if (parts.length === 0) return question;
  const head = parts.join("\n\n");
  return label ? `${head}\n\n---\n\n${label}${question}` : `${head}\n\n${question}`;
}
