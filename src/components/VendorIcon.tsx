// Official vendor logos for the model store: each author's real Hugging Face
// avatar (the logo the org uploaded itself), fetched once per author and
// cached. Offline or avatar-less authors fall back to the official Hugging
// Face logo, which ships with the app — nothing is hand-drawn.
import { useEffect, useState } from "react";
import hfLogo from "../assets/huggingface_logo.svg";
import { hfAuthorAvatar } from "../lib/ipc";

const cache = new Map<string, Promise<string | null>>();

function avatarFor(author: string): Promise<string | null> {
  let p = cache.get(author);
  if (!p) {
    p = hfAuthorAvatar(author).catch(() => null);
    cache.set(author, p);
  }
  return p;
}

export function OrgAvatar({ author, size = 20 }: { author: string; size?: number }) {
  const [url, setUrl] = useState<string | null>(null);
  const [broken, setBroken] = useState(false);
  useEffect(() => {
    let live = true;
    setUrl(null);
    setBroken(false);
    void avatarFor(author).then((u) => {
      if (live) setUrl(u);
    });
    return () => {
      live = false;
    };
  }, [author]);
  if (url && !broken) {
    return (
      <img
        className="org-avatar-img"
        src={url}
        width={size}
        height={size}
        alt=""
        loading="lazy"
        onError={() => setBroken(true)}
      />
    );
  }
  return <img className="org-avatar-img" src={hfLogo} width={size} height={size} alt="" />;
}
