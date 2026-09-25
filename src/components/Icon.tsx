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
  // The two modes, on their switch.
  chat: "M5 5.5h14a1.5 1.5 0 011.5 1.5v8.5a1.5 1.5 0 01-1.5 1.5H10l-4.5 3.5V17H5a1.5 1.5 0 01-1.5-1.5V7A1.5 1.5 0 015 5.5z",
  code: "M8.5 7.5L4 12l4.5 4.5M15.5 7.5L20 12l-4.5 4.5M13.2 5l-2.4 14",
  image: "M5 4.5h14A1.5 1.5 0 0120.5 6v12a1.5 1.5 0 01-1.5 1.5H5A1.5 1.5 0 013.5 18V6A1.5 1.5 0 015 4.5zM3.5 16l5-5 4 4 2.5-2.5 5 5M15.5 9.5a1.5 1.5 0 100-.01",
  // Auto-approve: runs without asking.
  bolt: "M13.5 3L5.5 13.5h6L10.5 21l8-10.5h-6L13.5 3z",
  refresh: "M20 12a8 8 0 11-2.34-5.66L20 8.5M20 3.5v5h-5",
  eject: "M12 5l7 8H5l7-8zM5 18.5h14",
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
