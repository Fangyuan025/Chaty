/** Lines a change added and removed, as every count of one shows them — a
 *  step's card, the folded run, the turn's changes card: only the parts that
 *  are not zero, with a real minus sign. A red "-0" beside every edit that
 *  removed nothing was noise; nothing at all shows when nothing changed. */
export function DiffStat({ added, removed }: { added: number; removed: number }) {
  if (!added && !removed) return null;
  return (
    <>
      {added ? <em className="plus">+{added}</em> : null}
      {removed ? <em className="minus">−{removed}</em> : null}
    </>
  );
}
