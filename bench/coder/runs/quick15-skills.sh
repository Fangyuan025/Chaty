#!/bin/bash
# M3 A/B: the quick15 subset with official skills on (CHATY_BENCH_SKILLS=1),
# one fresh runner process per task (campaign rule). Resume-safe.
set -u
RUNS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$RUNS/../../.." && pwd)"
SIDE="${1:-skills}"          # skills | noskills
LOG="$RUNS/quick15-$SIDE.log"
OUT="$RUNS/quick15-$SIDE-results.jsonl"
export CHATY_BENCH_MODEL="$HOME/Library/Application Support/com.chaty.desktop/models/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-MLX-mxfp8"
[ "$SIDE" = "skills" ] && export CHATY_BENCH_SKILLS=1
cd "$REPO"
python3 -c "
import json
q=json.load(open('bench/coder/quick15.json'))
[print(t) for k,v in q.items() if isinstance(v,list) for t in v]" | while IFS= read -r task; do
  [ -z "$task" ] && continue
  grep -q "\"task\":\"$task\"" "$OUT" 2>/dev/null && continue
  MEM=$(vm_stat | awk '/Pages free/{f=$3} /Pages wired/{w=$4} END{printf "free=%.1fG wired=%.1fG", f*16384/1e9, w*16384/1e9}')
  START=$(date +%s)
  echo "=== $(date '+%m-%d %H:%M:%S') START $task [$MEM]" >> "$LOG"
  perl -e 'alarm 2400; exec @ARGV' -- npx tsx bench/coder/runner.mts --tasks bench/coder/swebench/tasks --only "$task" >> "$LOG" 2>&1
  RC=$?
  # Only accept a row from a file the runner wrote DURING this task — an
  # `ls -t` without the freshness check happily folds in a stale row from an
  # earlier campaign and reports a pass that never happened (observed).
  NEWEST=$(find "$RUNS" -name '2026-*.jsonl' -newermt "@$START" 2>/dev/null | head -1)
  ROW=$([ -n "$NEWEST" ] && grep "\"task\":\"$task\"" "$NEWEST" | tail -1)
  if [ -n "$ROW" ]; then echo "$ROW" >> "$OUT"; else echo "{\"task\":\"$task\",\"resolved\":false,\"why\":\"runner died rc=$RC\"}" >> "$OUT"; fi
  R=$(echo "$ROW" | grep -o '"resolved":true')
  echo "$([ -n "$R" ] && echo ✓ || echo ✗) $task (rc=$RC)" >> "$LOG"
done
touch "$RUNS/quick15-$SIDE.done"
echo "=== QUICK15-$SIDE ALL DONE" >> "$LOG"
