---
name: mac-app
description: Build a runnable macOS .app the incremental way — feature, verify, next feature — and package + launch-check before delivering
when: the user asks for a macOS desktop app (mac 桌面应用/日历/记事本/工具类 GUI) — the deliverable must be a packaged .app that actually launches, not just source files
---

The deliverable is a **double-clickable .app that launches and works** — source
that merely compiles is half the job. Work in this exact rhythm:

1. **Skeleton first, verify immediately.** Scaffold the smallest thing that
   builds AND launches (empty window is fine). Package it, launch it, kill it.
   Only then start features.
2. **One feature at a time.** Implement → `validate_change` (or the build
   command) → green → next feature. Never stack 4+ unverified files.
3. **Never break green.** After a red result, suspect ONLY what you changed
   since the last green check; restore it if needed. Do not "improve" code
   that already passed — no renames, no restructuring, no style edits of
   working files.

## SwiftUI + SwiftPM (recommended for native)

Layout: `Package.swift` + `Sources/<Name>/…` with an `@main` SwiftUI `App`.
Build, package, and launch-check:

```bash
swift build -c release 2>&1 | tail -5
APP="<Name>.app"; BIN=".build/release/<Name>"
mkdir -p "$APP/Contents/MacOS" && cp "$BIN" "$APP/Contents/MacOS/<Name>"
printf '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict><key>CFBundleExecutable</key><string><Name></string><key>CFBundleIdentifier</key><string>local.<name></string><key>CFBundleName</key><string><Name></string><key>CFBundlePackageType</key><string>APPL</string><key>NSHighResolutionCapable</key><true/></dict></plist>' > "$APP/Contents/Info.plist"
"$APP/Contents/MacOS/<Name>" & APP_PID=$!; sleep 4
kill -0 $APP_PID && echo "LAUNCH OK (alive after 4s)" || echo "LAUNCH FAILED — fix before delivering"
kill $APP_PID 2>/dev/null
```

`LAUNCH FAILED` usually means a crash at startup — run the binary in
foreground briefly to read the crash text.

## Electron / web-view stack

`electron-builder --dir` (never full dmg — slow) puts the bundle under
`dist/mac*/<Name>.app`; launch-check the same way with
`dist/mac*/<Name>.app/Contents/MacOS/<Name>`. Remember: production must
`loadFile('dist/index.html')` — verify AFTER `vite build`, not against the
dev server only.

## Verification truths (don't learn these the hard way)

- **Browser screenshot ≠ system screenshot.** `browser_screenshot` captures
  the embedded browser's web page only — it can never show a native window.
  Native GUI proof = the launch + stay-alive check above.
- **"Operation not permitted" is usually NOT the sandbox.** The sandbox only
  blocks writes outside the workspace. Screen recording (`screencapture`)
  and app automation (`osascript` → other apps) fail on macOS *privacy
  authorization* (TCC) — do not abandon the task or switch stacks over it;
  verify without that permission instead.
- **One stack, start to finish.** If you must switch stacks, say why and
  delete the old implementation — never leave two half-apps in the tree.

## Delivery checklist (all four, in the answer)

- build green (real build, not `-parse`/`--version`)
- `.app` bundle exists in the workspace
- launch check printed `LAUNCH OK`
- features you claim were each verified when added
