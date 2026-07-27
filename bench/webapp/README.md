# ChatyWebapp-Bench

Four scenarios reproducing the four audited webapp-flow failure modes:

| id | 痛点 | objective grade |
|---|---|---|
| `server-start` | foreground dev server stalls the loop, then the timeout kills it | answer names the URL **and** the server is still reachable at turn end |
| `console-fix` | never reads the JS console → fixes the wrong thing | source actually fixed **and** a browser action happened after the last edit |
| `todo-follow` | update_plan once, then the plan is decoration | all three requested changes present in the files |
| `ship-verified` | delivers without walking the page | both edits present **and** browser action after the last edit |

Grading never trusts the model's self-report: file contents, an end-of-run
HTTP probe against the dev server, and step-log facts only.

## Run

```bash
CHATY_BENCH_MODEL=/path/to/model npx tsx bench/webapp/runner.mts \
    [--repo /path/to/tree] [--only id] [--tag label]
```

A/B is by **code side**: `--repo` selects which tree's `agentLoop.ts` runs and
(unless `CHATY_HEADLESS_BIN` overrides) which tree's release binary serves the
backend. Model, scenarios and grading stay identical across sides. Results go
to `runs/*.jsonl` (gitignored, per bench policy).

Scenario projects are offline-only (`python3 -m http.server` as the dev
script) — no npm install, no network variance.
