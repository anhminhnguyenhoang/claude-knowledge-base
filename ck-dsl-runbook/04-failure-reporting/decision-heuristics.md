---
name: decision-heuristics
description: Runbook S16: lever-direction heuristics (if X regresses, try Y) and the anti-pattern list (chasing flags before ISA, trusting one run, promoting from a smoke set, removing a cost without checking where it moved).
source: ck-dsl-optimization-runbook.md (lines 1971-2009)
---

## 16. Decision Heuristics

### 16.1 Lever-direction heuristics

- If the output dimension is smaller than MFMA width, consider direct
  / specialized mapping (`DirectConv4cSpec`, `4x4x4` atom).
- If an epilogue uses scalar stores, vectorize it before deeper
  changes.
- If global-to-LDS uses register intermediate, try direct async copy
  (`pipeline = "compv4"`).
- If barriers dominate, increase work per barrier or reduce stages.
- If wider tiles slow down, inspect extra LDS passes and register
  pressure.
- If smaller workgroups speed up, occupancy was likely limiting.
- If smaller workgroups slow down, data sharing / barrier amortization
  may dominate.
- If swizzle does not help through register staging, it may still
  matter only with the matching async distribution.
- If LDS epilogue slows down, direct vector stores may already be
  good enough.
- If compiler flags produce wrong answers, remove them even if
  another codebase uses them safely.
- If the HIP-debug backend disagrees with LLVM-direct on a kernel,
  one of the two is missing an op lowering — file the issue,
  don't ignore it.

### 16.2 Anti-patterns (what to avoid)

- Chasing compiler flags before inspecting IR / ISA.
- Trusting one benchmark run.
- Comparing kernels with different math or masks.
- Comparing compile + launch time to warm launch time.
- Ignoring failed correctness because the speed number is attractive.
- Adding compatibility shims for unshipped experimental branches
  instead of fixing the builder.
- Promoting a kernel variant from a smoke set without a full-cohort
  sweep (§12.2, §17.4).
- Removing one cost without checking whether it moved to a worse
  place in the per-iter ISA composition (§9.3, §17.4).

---
The one-page decision tree: [diagnostic-decision-tree](../06-reference/diagnostic-decision-tree.md). Anti-patterns proven in: [case-studies-overview](../05-case-studies/case-studies-overview.md).
