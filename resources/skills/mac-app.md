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

## SwiftUI + SwiftPM (the ONLY supported scaffold here)

**Never hand-write an `.xcodeproj`.** A hand-made `project.pbxproj` is
almost always malformed — unreadable by xcodebuild, wrong file references,
broken resource paths (this audit hit all three). SwiftPM gives you build,
tests, and packaging with zero project files; editing an .xcodeproj is only
for projects that already ship one.

Layout: `Package.swift` + `Sources/<Name>/…` with an `@main` SwiftUI `App`.
Declare the platform UP FRONT — `platforms: [.macOS(.v14)]` — modern SwiftUI
APIs (`ContentUnavailableView`, …) are macOS 14+, and an availability error
discovered late forces a rewrite you have no budget for. If one still
appears, RAISE the platform version; never rewrite working views around it.
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

## SwiftUI patterns that break small models (copy these, don't improvise)

**Editable list items** — inside `List`/`ForEach` the element is a VALUE
copy; you cannot make a `Binding` from it. Iterate the binding collection:

```swift
List { ForEach($store.notes) { $note in
    TextField("标题", text: $note.title)   // $note IS a Binding
} .onDelete { store.notes.remove(atOffsets: $0) } }
```

**Master–detail editor** — select by id, edit through a computed binding:

```swift
List(store.notes, selection: $selectedID) { note in Text(note.title) }
if let i = store.notes.firstIndex(where: { $0.id == selectedID }) {
    NoteEditor(note: $store.notes[i])
}
```

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

## Functional bar: launching is not "working"

Structure for testability from the first file: **core logic in plain types**
(state machines, stores, calculations — no SwiftUI imports), views thin.
Add a test target and exercise every basic function:

```
Tests/<Name>Tests/CoreTests.swift   // import XCTest + @testable import <Name>
swift test 2>&1 | tail -3           // must end "0 failures"
```

**Use THREE targets** — tests must depend on a library, never on the
executable (linking an `@main` target into tests duplicates `_main`):

```swift
// swift-tools-version:5.9
import PackageDescription
let package = Package(
    name: "<Name>", platforms: [.macOS(.v14)],
    targets: [
        .target(name: "<Name>Core"),                                // logic — testable
        .executableTarget(name: "<Name>", dependencies: ["<Name>Core"]), // @main + views
        .testTarget(name: "<Name>CoreTests", dependencies: ["<Name>Core"]),
    ]
)
```

Two rules make the split work — missing either is the #1 multi-target
failure ("cannot find 'X' in scope"):
1. **Everything in Core is `public`**: `public struct Note`, `public class
   NoteStore`, `public init(...)`, `public func …`, `@Published public var …`.
2. **Every file in the executable target starts with `import <Name>Core`.**
Test the real flows: add → edit → delete → reload/persist round-trip, plus
one edge case each (empty input, missing file). A feature without a passing
test or an executed proof is not done.

**After your LAST source edit, the packaging pipeline runs AGAIN** —
rebuild (`swift build -c release`; `swift test` only refreshes DEBUG
products, never the release binary) → re-copy into the `.app` → re-run the
launch check. An `.app` packaged before your final fix ships the bug you
just fixed.

## Delivery checklist (all five, in the answer)

- build green (real build, not `-parse`/`--version`)
- `swift test` green with the core functions covered
- `.app` bundle exists in the workspace
- launch check printed `LAUNCH OK`
- each claimed feature names its proof (test name or executed command)
