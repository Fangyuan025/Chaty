import type React from "react";
// The app's single icon system: one stroke family, one default weight, sized
// by prop. UI chrome must use <Icon> (or an inline SVG matching this spec) —
// never text glyphs (✓ ▸ ×) whose baseline/weight drift with the font.

const PATHS: Record<string, string> = {
  x: "M5 5l14 14M19 5L5 19",
  check: "M4.5 12.5l5 5L19.5 7",
  plus: "M12 5v14M5 12h14",
  "chevron-down": "M6 9.5l6 6 6-6",
  "chevron-right": "M9.5 6l6 6-6 6",
  ban: "M12 3a9 9 0 100 18 9 9 0 000-18zM5.8 5.8l12.4 12.4",
  pin: "M12 3l1.8 5.4H19l-4.3 3.4 1.7 5.5-4.4-3.4-4.4 3.4 1.7-5.5L5 8.4h5.2z",
  search: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-4.5-4.5",
  download: "M12 3v12M7 10l5 5 5-5M5 21h14",
  // Folder — CodeMode drew this same path inline three times.
  folder: "M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z",
  lines: "M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01",
};

export function Icon({
  name,
  size = 14,
  strokeWidth = 1.8,
  className,
  style,
}: {
  name: keyof typeof PATHS | string;
  size?: number;
  strokeWidth?: number;
  className?: string;
  /** Inline overrides (alignment nudges) — icons.tsx wrappers pass it through. */
  style?: React.CSSProperties;
}) {
  return (
    <svg
      className={className}
      style={style}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name] ?? PATHS.x} />
    </svg>
  );
}
