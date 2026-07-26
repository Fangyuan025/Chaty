import { Icon } from "./Icon";
// Small, consistent stroke icons — replace emoji across the UI so nothing
// platform-renders an inline picture-glyph that breaks the visual grade.
// All inherit color + scale from the parent (currentColor, 1.7 stroke).
//
// Division of labour with Icon.tsx: single-stroke glyphs live there as PATHS
// entries; this file is for multi-element or filled art that a single path
// can't express (play/stop triangles, filled pin). A glyph that exists in
// both must delegate — see IconSearch/IconDownload below.

function Svg({
  size = 16,
  style,
  children,
}: {
  size?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ flex: "none", verticalAlign: "-0.18em", ...style }}
    >
      {children}
    </svg>
  );
}

type P = { size?: number; style?: React.CSSProperties };

/** Knowledge base — a book. */
export const IconKb = ({ size, style }: P) => (
  <Svg size={size} style={style}>
    <path d="M5 4.5A1.5 1.5 0 0 1 6.5 3H19a1 1 0 0 1 1 1v13.5" />
    <path d="M6.5 16.5H20v2.5a1 1 0 0 1-1 1H6.5A1.5 1.5 0 0 1 5 18.5V4.5" />
  </Svg>
);

/** A document / file. */
export const IconDoc = ({ size, style }: P) => (
  <Svg size={size} style={style}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" />
  </Svg>
);

/** Microphone / voice. */
export const IconMic = ({ size, style }: P) => (
  <Svg size={size} style={style}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </Svg>
);

/** Magnifier / search. */
// search and download exist in Icon.tsx's PATHS too, and the two download
// copies had already drifted apart (M7 11 here vs M7 10 there — the same
// arrow, one pixel different, in one app). These delegate so each shape has
// exactly ONE definition; the wrappers stay for the call sites' sake.
export const IconSearch = ({ size, style }: P) => <Icon name="search" size={size} style={style} />;

/** Download / export. */
export const IconDownload = ({ size, style }: P) => <Icon name="download" size={size} style={style} />;

/** Refresh / regenerate. */
export const IconRefresh = ({ size, style }: P) => (
  <Svg size={size} style={style}>
    <path d="M21 12a9 9 0 1 1-2.6-6.3M21 4.5V9h-4.5" />
  </Svg>
);

/** Play (filled triangle). */
export const IconPlay = ({ size = 16, style }: P) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
    style={{ flex: "none", verticalAlign: "-0.18em", ...style }}
  >
    <path d="M7 4.5v15l13-7.5z" />
  </svg>
);

/** Stop (filled square). */
export const IconStop = ({ size = 16, style }: P) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
    style={{ flex: "none", verticalAlign: "-0.18em", ...style }}
  >
    <rect x="6" y="6" width="12" height="12" rx="2.5" />
  </svg>
);

/** Pin (outline) — for unpinned conversations. */
export const IconPin = ({ size, style }: P) => (
  <Svg size={size} style={style}>
    <path d="M9 4h6l-1 5 3 3v2H7v-2l3-3-1-5zM12 14v6" />
  </Svg>
);

/** Pin (filled) — for pinned conversations. */
export const IconPinFilled = ({ size = 16, style }: P) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    stroke="currentColor"
    strokeWidth={1.2}
    strokeLinejoin="round"
    aria-hidden="true"
    style={{ flex: "none", verticalAlign: "-0.18em", ...style }}
  >
    <path d="M9 4h6l-1 5 3 3v2H7v-2l3-3-1-5z" />
    <path d="M12 14v6" fill="none" />
  </svg>
);

/** Edit / rename — a pencil. */
export const IconEdit = ({ size, style }: P) => (
  <Svg size={size} style={style}>
    <path d="M14.5 5.5l4 4M4 20.5l1-4L16 5.5a1.8 1.8 0 0 1 2.5 0l.5.5a1.8 1.8 0 0 1 0 2.5L8 19.5l-4 1z" />
  </Svg>
);

/** Deep research — a lab flask. */
export const IconResearch = ({ size, style }: P) => (
  <Svg size={size} style={style}>
    <path d="M9.5 3h5M10.5 3v6.2L5.6 17.6A2 2 0 0 0 7.3 21h9.4a2 2 0 0 0 1.7-3.4L13.5 9.2V3" />
    <path d="M8 14.5h8" />
  </Svg>
);
