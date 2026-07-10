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
};

export function Icon({
  name,
  size = 14,
  strokeWidth = 1.8,
  className,
}: {
  name: keyof typeof PATHS | string;
  size?: number;
  strokeWidth?: number;
  className?: string;
}) {
  return (
    <svg
      className={className}
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
