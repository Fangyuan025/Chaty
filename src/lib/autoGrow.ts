/** Height a composer should take to show all of its content.
 *
 *  A one-row textarea does not grow: its box stays a single line and anything
 *  taller — a wrapped line, or just a long placeholder in a narrow box —
 *  scrolls instead. Raising the UI scale narrows the layout, the placeholder
 *  wraps, and an empty composer answers with scrollbars and half a line of
 *  text. So the box follows its content, up to a ceiling.
 */
export function contentHeight(scrollHeight: number, maxPx: number): number {
  return Math.min(scrollHeight, maxPx);
}

/** Size `el` to its content. Measuring needs the height released first —
 *  `scrollHeight` never shrinks below the height already set — and writing
 *  back the height it already had is left as a no-op, so an observer watching
 *  this element's box settles instead of ringing forever. */
export function fitToContent(el: HTMLTextAreaElement, maxPx: number): void {
  const had = el.style.height;
  el.style.height = "auto";
  const want = `${contentHeight(el.scrollHeight, maxPx)}px`;
  el.style.height = want;
  if (want === had) return;
}

/** Watch `box` and keep `el` sized to its content: the composer has to
 *  re-measure whenever the row's WIDTH changes (window resize, UI scale,
 *  sidebar toggle), not only when the text does. Observing the row rather
 *  than the textarea keeps the height we set from feeding back in. */
export function watchContentHeight(
  el: HTMLTextAreaElement,
  box: Element,
  maxPx: number,
): () => void {
  const fit = () => fitToContent(el, maxPx);
  fit();
  // The first measurement happens before the interface font has loaded, and
  // the fallback wraps differently: a placeholder that took one line in the
  // fallback takes two in the real font. No element's width changes when the
  // font arrives, so nothing else would ask for a second look — and the
  // composer would keep the height it computed against a font it is no longer
  // showing, scrolling the line that no longer fits.
  void document.fonts?.ready.then(fit).catch(() => {});
  // Two triggers, because neither covers everything. The observer catches the
  // row changing width on its own (a sidebar opening, a panel resizing). The
  // window event catches a change of UI scale, which is what put a wrapped
  // placeholder in a one-line box to begin with: page zoom resizes the
  // viewport, and every box in it, without any element resizing "by itself".
  // WIDTH only, and this is not a nicety: sizing the box changes the row's
  // HEIGHT, the observer reports that back, and the loop never settles —
  // WebKit starts dropping notifications and the composer keeps whatever
  // height it happened to have when the drops began. Its own height changes
  // tell this function nothing; only a change of available width does.
  let seenWidth = box.getBoundingClientRect().width;
  const ro = new ResizeObserver(() => {
    const width = box.getBoundingClientRect().width;
    if (width === seenWidth) return;
    seenWidth = width;
    fit();
  });
  ro.observe(box);
  window.addEventListener("resize", fit);
  return () => {
    ro.disconnect();
    window.removeEventListener("resize", fit);
  };
}
