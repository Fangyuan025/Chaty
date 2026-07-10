import { useEffect, useState } from "react";

/**
 * Exit-transition helper: keeps a component mounted for `ms` after `open`
 * flips false so its closing animation can play. Pair with the `.closing`
 * class (ui-pop-out / ui-fade-out in App.css).
 */
export function useExitTransition(open: boolean, ms = 150): { mounted: boolean; closing: boolean } {
  const [mounted, setMounted] = useState(open);
  const [closing, setClosing] = useState(false);

  useEffect(() => {
    if (open) {
      setMounted(true);
      setClosing(false);
      return;
    }
    if (!mounted) return;
    setClosing(true);
    const timer = setTimeout(() => {
      setMounted(false);
      setClosing(false);
    }, ms);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, ms]);

  return { mounted, closing };
}
