/**
 * Localization coverage report:  npx tsx scripts/l10n-status.mts [locale]
 *
 * Lists, per community locale, how many of the UI strings in src/lib/i18n.tsx
 * are translated and which keys are still falling back to English. zh/en are
 * the native pair and always complete (the compiler enforces them).
 */
import { LANGS, T } from "../src/lib/i18n";

const keys = Object.keys(T) as (keyof typeof T)[];
const community = LANGS.map((l) => l.id).filter((id) => id !== "zh" && id !== "en");
const only = process.argv[2];

for (const locale of community) {
  if (only && locale !== only) continue;
  const missing = keys.filter((k) => !(T[k] as Record<string, string | undefined>)[locale]);
  const done = keys.length - missing.length;
  console.log(`${locale}: ${done}/${keys.length} translated (${((done / keys.length) * 100).toFixed(1)}%)`);
  if (missing.length) {
    console.log(`  missing (fall back to English):`);
    for (const k of missing) console.log(`    ${k}`);
  }
}
if (!community.length) console.log("no community locales declared in LANGS yet");
