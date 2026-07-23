# ChatyWeb-Bench

A WebArena-style benchmark for Chaty's **browser agent** — the `browser_*`
tool chain — scored end to end against local, deterministic web fixtures.
Everything runs on one machine: the real production agent loop
(`src/lib/agentLoop.ts`), the real Rust tool layer (`agent.rs` → `browser.rs`),
a real headless Chrome, and a local fixture server. No network, no flakiness,
no external accounts.

## Why not WebArena itself?

WebArena's self-hosted sites (shopping, GitLab, Reddit clones) are excellent
but heavy — multi-container Docker deployments that don't fit a
laptop-under-a-local-35B workflow, and their tasks assume a vision-capable
agent. ChatyWeb-Bench borrows the *shape* (self-hosted deterministic sites,
programmatic graders, task categories spanning navigation, extraction, forms,
and multi-step transactions) at local-first scale. Scores are **not**
comparable to WebArena leaderboard numbers.

## Layout

| Piece | What it is |
| --- | --- |
| `server.mts` | Zero-dependency fixture server: six apps + JSON state API. The server is the single source of truth — every UI mutation goes through `POST /api/...`, so graders read `getState()` and never scrape the DOM. `POST /api/reset` restores the seeded world. |
| `sites/*.html` | Six self-contained single-file apps: **shop** (catalog/cart/checkout), **inbox** (read/reply/archive/compose), **board** (kanban), **wiki** (hash-routed handbook), **forms** (two-step wizard with server-side validation), **admin** (paginated/sortable/filterable user table with row editing). |
| `tasks.json` | 23 tasks. `type: "state"` tasks are graded on server state; `type: "answer"` tasks on the agent's final message. Each task carries an **oracle**: a known-good action sequence through the real tool chain. |
| `graders.mts` | One grader per task: state assertions (including collateral-damage checks) or required-substring answer matching. |
| `oracle.mts` | Gold validation, no model: replay every oracle through chaty-headless → browser.rs → headless Chrome, then grade. A task that can't pass its own oracle never enters a scored run. Gate: **23/23**. |
| `runner.mts` | The scored run: real `runAgentTurn` per task, fixture reset before, grade after, JSONL row out. |

## Text-browser mode

The bench model (Qwen3.5-35B-A3B — MoE, ~3B active per token) has **no vision
encoder**. The run uses `browserTextMode: true`: the agent gets the browser
suite minus the two screenshot tools, and `browser_read`'s rich digest
(visible text + interactive elements + current input values) is its only eyes.
That makes this a benchmark of Chaty's *text-first* browser UX — exactly the
surface a local-model product must get right.

## Budgets & parameters

One task = fresh headless process, fresh Chrome, fresh fixture state.
30 steps/turn, auto-continue once (≤ 2 turns), temperature 0.2, think off,
nCtx 16384 — the coder-bench envelope adapted to shorter web tasks.

## Run

```bash
# 1. build the tool server
cargo build --bin chaty-headless   # in src-tauri/

# 2. validate fixtures + graders (no model, ~1 min)
npx tsx bench/web/oracle.mts

# 3. scored run
CHATY_BENCH_MODEL=/path/to/model npx tsx bench/web/runner.mts [--only task-id]
```

Env: `CHATY_HEADLESS_BIN` (defaults to `target/debug`), `CHATY_BENCH_TRANSCRIPT=<dir>`
for per-task transcripts, `CHATY_WEBBENCH_PORT` (default 8763).

Results land in `bench/web/runs/*.jsonl` (gitignored — cite the harness commit
instead).
