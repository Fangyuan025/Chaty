/** Where a send waits: the model is at work elsewhere — a Code session, a
 *  chat reply — and one engine does one thing at a time. Says where, a light
 *  passing over the words while it lasts, and takes you there. Over a
 *  composer (`dock`) it is a tab tucked behind the box's top edge, centred on
 *  it; the box still takes typing, only sending waits. */
export function BusyNote({ text, go, onGo, dock }: { text: string; go?: string; onGo?: () => void; dock?: boolean }) {
  return (
    <div className={`busy-note${dock ? " busy-dock" : ""}`} role="status">
      <span className="busy-note-text" title={text}>
        {text}
      </span>
      {go && onGo && (
        <button type="button" className="busy-note-go" onClick={onGo}>
          {go}
        </button>
      )}
    </div>
  );
}
