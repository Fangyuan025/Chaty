/** The round "back to the latest" arrow over a scrolling thread. It stays in
 *  the page and fades — in when the thread is scrolled up, out when it is back
 *  at the bottom — instead of popping in and out of existence; hidden, it
 *  takes no clicks and no keyboard focus. */
export function JumpBottom({ shown, title, onClick }: { shown: boolean; title: string; onClick: () => void }) {
  return (
    <button
      className={`jump-bottom${shown ? " shown" : ""}`}
      title={title}
      onClick={onClick}
      tabIndex={shown ? 0 : -1}
      aria-hidden={!shown}
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  );
}
