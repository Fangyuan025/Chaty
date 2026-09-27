import { useI18n } from "../lib/i18n";
import { tl, type ParamDef } from "../lib/musicGen";
import { Select } from "./Select";

/** How full a slider is, for the track's fill (the app's slider style). */
export function fill(v: number, min: number, max: number): React.CSSProperties {
  const pct = max > min ? ((v - min) / (max - min)) * 100 : 0;
  return { ["--fill" as string]: `${Math.max(0, Math.min(100, pct))}%` };
}

/** One of a family's parameters: its control, showing the value it runs
 *  with, and — once moved off it — a way back to the recommendation. */
export function MusicParamField({
  p,
  value,
  custom,
  onChange,
  compact,
}: {
  p: ParamDef;
  /** The value it runs with (the setting, else the recommendation). */
  value: string;
  /** Moved off the recommendation. */
  custom: boolean;
  /** null = back to the recommendation. */
  onChange: (v: string | null) => void;
  /** The composer's quick panel (narrower). */
  compact?: boolean;
}) {
  const { t, lang } = useI18n();
  const label = tl(p.label, lang);
  const tip = p.tip ? tl(p.tip, lang) : undefined;
  const rec = p.defLabel ? tl(p.defLabel, lang) : p.def === "" ? t("musAuto") : p.def;
  const head = (
    <span className="ms-field-head">
      <span className="ms-field-label" title={tip}>
        {tip ? <em className="has-tip" data-tip={tip}>{label}</em> : label}
        {(p.kind === "int" || p.kind === "float") && value !== "" && <b>{value}</b>}
      </span>
      {custom ? (
        <button type="button" className="ms-field-reset" onClick={() => onChange(null)} title={t("musResetOne")}>
          {t("musRecommendedShort")} {rec}
        </button>
      ) : (
        <em className="ms-field-rec">{t("musRecommended")}</em>
      )}
    </span>
  );

  let control: React.ReactNode;
  if (p.kind === "enum") {
    control = (
      <Select
        value={value}
        ariaLabel={label}
        onChange={(v) => onChange(v === p.def ? null : v)}
        options={(p.choices ?? []).map((c) => ({ value: c.value, label: c.label ? tl(c.label, lang) : c.value }))}
      />
    );
  } else if (p.kind === "bool") {
    const on = value === "true" || value === "1";
    control = (
      <label className="toggle-row ms-toggle">
        <input type="checkbox" checked={on} onChange={(e) => onChange(String(e.target.checked) === p.def ? null : String(e.target.checked))} />
        <span>{on ? t("musOn") : t("off")}</span>
      </label>
    );
  } else if (p.kind === "text") {
    control = (
      <input
        type="text"
        className="ms-text"
        value={custom ? value : ""}
        placeholder={p.def || t("musAuto")}
        onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
      />
    );
  } else {
    const n = Number(value);
    const has = value !== "" && Number.isFinite(n);
    const min = p.min ?? 0;
    const max = p.max ?? Math.max(1, has ? n * 4 : 100);
    const step = p.step ?? (p.kind === "int" ? 1 : 0.01);
    const bounded = p.min != null && p.max != null && p.max - p.min <= 10000;
    const set = (v: number) => {
      const s = p.kind === "int" ? String(Math.round(v)) : String(Math.round(v / step) * step).replace(/(\.\d*?)0{6,}\d*$/, "$1");
      onChange(Number(s) === Number(p.def) && p.def !== "" ? null : s);
    };
    control = (
      <span className="ms-num">
        {bounded && (
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={has ? n : min}
            style={fill(has ? n : min, min, max)}
            onChange={(e) => set(Number(e.target.value))}
          />
        )}
        <input
          type="number"
          min={p.min}
          max={p.max}
          step={step}
          value={has ? value : ""}
          placeholder={p.def === "" ? t("musAuto") : undefined}
          onChange={(e) => (e.target.value === "" ? onChange(null) : set(Number(e.target.value)))}
        />
      </span>
    );
  }

  return (
    <div className={`ms-field ${compact ? "compact" : ""} ${custom ? "custom" : ""}`}>
      {head}
      {control}
    </div>
  );
}
