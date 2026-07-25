# ChatyMCP-Bench

Does an MCP server's tools actually work **in the hands of a small local
model**? Connecting is table stakes — the store's "certified" badge covers
that (`cargo test store_cert`). This bench answers the harder question, and
its answer is what the badge is worth.

The tasks run the real agent loop (`runAgentTurn`) against real MCP servers
started by the real client (`mcp.rs`), with tools registered through the real
registry — same end-to-end posture as ChatyWeb-Bench, one layer up.

## Grading

Server state is the source of truth, never the transcript:

- **state** tasks read the server back through a *different* tool than the one
  under test (write with `create_entities`, verify with `read_graph`), so a
  model that narrates success without acting fails.
- **answer** tasks match a normalized substring in the model's final text.

## Running

```bash
CHATY_BENCH_MODEL=/path/to/model npx tsx bench/mcp/runner.mts [--only task-id]
```

One fresh process per task; each task gets a clean temp state directory, so
runs are order-independent and resumable. Results go to `runs/` (gitignored,
like every other bench product).

## Why these tasks

The suite deliberately mixes the two failure modes that killed other agents on
SWE-bench: **tool-count pressure** (the filesystem server brings 14 tools, over
the 8-tool core-tier line, so it exercises the deferred index + full-doc
recovery path) and **schema literacy** (memory's `create_entities` takes a
nested array of objects — the shape small models most often flatten).
