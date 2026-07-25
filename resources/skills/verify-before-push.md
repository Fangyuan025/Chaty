---
name: verify-before-push
description: Prove a change works before it leaves the machine
when: the user asks to commit, push, or ship a change
---

Never push a change whose correctness you have only reasoned about.

1. **Pick the gate that can actually fail.** Type-checkers and linters catch
   shape errors, not behavior. If the change alters behavior, the gate is a
   test that fails without your fix and passes with it — write it if it
   doesn't exist.
2. **Run the gate and read the output.** Do not infer success from "no error
   printed"; find the pass count or the assertion that proves the new path ran.
3. **For anything visual or interactive**, capture evidence: a screenshot,
   a measured pixel value, a page digest. A description of what should happen
   is not evidence.
4. **Commit only after the gate is green**, and put what you verified in the
   commit message — the next reader needs to know which claims were checked.
5. **If a gate can't be run locally** (another OS, real credentials, a GPU),
   say so plainly and name what remains unverified. Do not present it as done.
