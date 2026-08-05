---
name: debug-playbook
description: Pick the RIGHT debug method for the failure you actually have — compile, crash, wrong output, dead UI — on any stack
when: a build fails, an app crashes or misbehaves, output is wrong, tests are red, or you are about to deliver an app whose functions you have not exercised
---

Match the method to the symptom. Re-running the same command and hoping is
never a method.

## 1. Compile / build error
Read the FIRST `file:line` in the output → `read_file` that spot → fix that
one error → rebuild. Repeat one error at a time. Never re-run an unchanged
build; never "fix" files the error doesn't name.

## 2. Crash at runtime
Run the thing in the foreground and read the top stack frame that is in
YOUR code (not the framework). Crash on launch with no output → run the
binary directly (not via `open`) to see stderr.

## 3. Wrong behavior / logic bug
Write a minimal failing test FIRST — one that states the expected result —
then fix until it passes. Keep the test. This is faster than print-guessing,
and it becomes your functional proof at delivery.

## 4. Per-stack functional verification (the delivery bar)
Compiling and launching is the entry ticket, not the goal — every basic
function must be EXECUTED before delivery:

- **CLI tool**: run it with real inputs for every subcommand/flag you
  implemented; check the actual output, including the error paths
  (missing file, bad args).
- **Web app**: start the dev server, `browser_navigate`, and CLICK through
  every feature — add/edit/delete/toggle, then reload to prove persistence.
  Watch `[console]` errors in tool results.
- **HTTP API**: `curl -f` every endpoint (or print `-w '%{http_code}'` and
  read it) — plain curl exits 0 on a 404, so without `-f` your
  "verification" verifies nothing. Cover happy path AND a 404/400 case,
  check the JSON body, and remember route params arrive as STRINGS —
  compare with `String(id)` / cast before `===`.
- **Native GUI (SwiftUI …)**: UI can't be clicked headlessly, so structure
  for testability: core logic (state, storage, calculations) lives in plain
  types, views stay thin. Write tests for that core and run `swift test`
  (SwiftPM layout: `Tests/<Name>Tests/`). Launch + stay-alive covers the
  shell; the tests cover the functions.
- **Library/module**: its test suite IS the functional bar — every public
  function gets at least one test.

## 5. When a command is DENIED
"Operation not permitted" on a build/cache path → workspace-relative paths;
screen/automation denials are macOS TCC, not the sandbox — verify another
way, never abandon the stack over it.

## Delivery rule
For each basic function, name in your final answer HOW it was verified
(test name, click path, or the exact command + output). A function with no
executed proof is not done. After verifying, CLEAN UP your test artifacts —
sample records, scratch files, seeded data — so the user receives a fresh
state, and note the cleanup in your answer.
