---
name: investigate-first
description: Understand an unfamiliar codebase before changing it
when: starting work in a repo or area you haven't touched yet
---

1. **Orient once**: `understand_repo` for the shape, then read the build/test
   commands. Knowing how to run the tests early is what makes every later step
   verifiable.
2. **Search by meaning, not by guess**: `search_code` with the behavior you're
   looking for ("where uploads are validated"), then `outline` the files it
   returns before reading them whole.
3. **Follow the data, not the names.** Find where the value you care about is
   produced and where it is consumed; the code in between is the part you
   actually need to understand.
4. **Match local conventions** — comment density, naming, error handling. A
   change that reads like the file around it needs no explanation.
5. **Check for existing tests of the area** before writing new ones; extend
   the established pattern rather than introducing a second style.
