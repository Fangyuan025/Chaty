import { useLayoutEffect, useRef, type HTMLAttributes } from "react";
import { prefersStill } from "../lib/smoothHeight";

/** A title that changes where it stands — a Code session named once its
 *  first message is in, a conversation once its first reply is — comes in
 *  with a short fade out of a slight blur instead of in a frame. The text it
 *  is first drawn with is simply there. The fade is on the element itself, so
 *  it carries the light passing over a busy title with it. */
export function SwapText({ text, ...rest }: { text: string } & HTMLAttributes<HTMLSpanElement>) {
  const ref = useRef<HTMLSpanElement>(null);
  const was = useRef(text);
  useLayoutEffect(() => {
    if (was.current === text) return;
    was.current = text;
    const el = ref.current;
    if (!el?.animate || prefersStill()) return;
    el.animate(
      [
        { opacity: 0, filter: "blur(4px)" },
        { opacity: 1, filter: "blur(0px)" },
      ],
      { duration: 520, easing: "cubic-bezier(0.22, 0.8, 0.24, 1)" },
    );
  }, [text]);
  return (
    <span ref={ref} {...rest}>
      {text}
    </span>
  );
}
