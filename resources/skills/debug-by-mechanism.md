---
name: debug-by-mechanism
description: Find the mechanism behind a bug instead of pattern-matching a fix
when: something is broken, failing, or behaving unexpectedly
---

A fix that makes the symptom disappear without explaining it usually moved the
bug somewhere quieter.

1. **Reproduce it smaller.** Strip the case until one more removal makes the
   symptom vanish. That boundary is where the mechanism lives.
2. **Read the code on the failing path** — the actual current source, not your
   memory of it. Note every place the observed value could come from.
3. **Name the mechanism in one sentence** before editing anything: "X happens
   because Y runs before Z." If you can't, you're still guessing.
4. **Predict something else that must be true** if your explanation is right,
   then check it. A wrong theory usually fails this step, cheaply.
5. **Fix the mechanism, not the symptom.** Then confirm the original case, and
   confirm a neighbouring case that shares the mechanism.
6. **If two attempts fail**, the mechanism is wrong — go back to step 2 rather
   than layering another guess on top.
