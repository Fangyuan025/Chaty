#!/bin/bash
# Mechanical verdict. Behaviour is checked directly rather than by running the
# agent's own tests through a `python3` that may not have pytest — a grader
# that fails on the grader's environment says nothing about the work.
set -e
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
from ledger import add_entry, total, largest, by_name, remove
e = []
add_entry(e, "a", 3); add_entry(e, "b", 5); add_entry(e, "a", 2)
assert total(e) == 10, f"total {total(e)}"
assert largest(e)["name"] == "b", largest(e)
assert len(by_name(e, "a")) == 2, by_name(e, "a")
assert by_name(e, "zzz") == [], by_name(e, "zzz")
assert remove(e, "a") == 2, "remove count"
assert remove(e, "zzz") == 0, "remove missing"
try:
    largest([])
except ValueError as ex:
    assert "empty ledger" in str(ex), str(ex)
else:
    raise AssertionError("largest([]) did not raise ValueError")
PY
# The agent was asked for tests and a CLI; both must exist and the CLI must run.
test -f tests/test_ledger.py
test -f cli.py
python3 cli.py total | grep -qE '[0-9]'
# Tests are run only where pytest is actually available.
if python3 -c "import pytest" 2>/dev/null; then python3 -m pytest -q tests/ >/dev/null; fi
