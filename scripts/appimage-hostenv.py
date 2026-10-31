"""Print the environment a command the agent runs sees inside the AppImage.

The image's start scripts are sourced the way AppRun sources them, and the
library and program paths AppRun.wrapped adds are added, so chaty-headless
(the app's agent layer) runs as it does in the app. It is then asked to run
`env` and where `xdg-open` is; scripts/appimage-check.sh fails if any of it
names the image (issue #28).

Usage: python3 -I scripts/appimage-hostenv.py <extracted image root> <workspace>
"""

import json
import os
import subprocess
import sys

root, workspace = sys.argv[1], sys.argv[2]

START = """
source "$APPDIR/apprun-hooks/linuxdeploy-plugin-gstreamer.sh"
source "$APPDIR/apprun-hooks/linuxdeploy-plugin-gtk.sh"
export PATH="$APPDIR/usr/bin/:$APPDIR/usr/sbin/:$PATH"
export LD_LIBRARY_PATH="$APPDIR/usr/lib/:$APPDIR/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH:-}"
export XDG_DATA_DIRS="$APPDIR/usr/share/:${XDG_DATA_DIRS:-}"
export APPIMAGE="$APPDIR/../Chaty.AppImage"
env -0
"""
raw = subprocess.run(
    ["bash", "-c", START], env={**os.environ, "APPDIR": root}, capture_output=True, check=True
).stdout
env = dict(kv.split("=", 1) for kv in raw.decode().split("\0") if "=" in kv)

app = subprocess.Popen(
    [os.path.join(root, "usr/bin/chaty-headless")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    env=env,
    text=True,
)


def ask(id, cmd, args):
    app.stdin.write(json.dumps({"id": id, "cmd": cmd, "args": args}) + "\n")
    app.stdin.flush()
    for line in app.stdout:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("id") != id:
            continue
        if msg.get("type") == "error":
            sys.exit(f"{cmd}: {msg.get('message')}")
        if msg.get("type") == "result":
            return msg["result"]
    sys.exit(f"{cmd}: chaty-headless ended without an answer")


ask(1, "agent_set_workspace", {"path": workspace})
ran = ask(2, "agent_bash", {"command": "env; echo xdg-open=$(command -v xdg-open)"})
app.stdin.close()
app.wait(timeout=30)
print(ran.get("stdout", ""))
if "xdg-open=" not in ran.get("stdout", ""):
    sys.exit(f"unexpected answer: {ran}")
