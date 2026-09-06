Work in this repository, step by step.

1. Read README.md and src/ledger.py so you know what is there.
2. Add these to src/ledger.py, keeping `add_entry` working:
   - `total(entries)` — sum of every amount.
   - `largest(entries)` — the entry with the biggest amount; raise `ValueError`
     with the message "empty ledger" when there are no entries.
   - `by_name(entries, name)` — every entry with that exact name, in order.
   - `remove(entries, name)` — drop every entry with that name and return how
     many were dropped.
3. Write tests in tests/test_ledger.py covering all five functions, including
   the ValueError case and the "nothing matched" case for by_name and remove.
4. Run the tests with `python3 -m pytest -q` and fix whatever fails until they
   all pass.
5. Add a `cli.py` at the repo root: `python3 cli.py total` prints the total of a
   hard-coded sample ledger of at least three entries. Run it and check the
   number it prints is right.
6. Update README.md to document all five functions and the CLI.

Do not stop until the tests pass and the CLI runs.
