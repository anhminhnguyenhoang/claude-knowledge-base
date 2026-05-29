---
name: diagnostic-decision-tree
description: Runbook Appendix: the one-page diagnostic decision tree (build smoke -> occupancy -> intrinsic -> targeted bench -> ISA -> rocprof) plus a complementary symptom-to-action table. Intentionally lossy — points at the next section to read.
source: ck-dsl-optimization-runbook.md (lines 3038-3133)
---

## Appendix: One-Page Diagnostic Decision Tree

```text
new kernel / regression
   │
   ▼
1) Build smoke (probe_config_sweep --only-build)
      │
      ├─ SPEC-FAIL → fix coupled fields (validation in spec __post_init__)
      ├─ BUILD-FAIL → reduce / isolate failing variant; verify IR coverage
      └─ all build  → 2)
   │
   ▼
2) probe_occupancy
      │
      ├─ spill > 0     → reduce VGPR pressure (atom, accumulators, unroll)
      ├─ waves < 4     → check LDS budget; try pipeline="lean"; smaller tile_k
      ├─ limited LDS   → recheck swizzle (Section 6.4a); smaller tile or async
      └─ healthy       → 3)
   │
   ▼
3) probe_intrinsic_counts
      │
      ├─ no MFMA       → wrong atom selected; check Section 7
      ├─ no async DMA  → pipeline != "compv4" (or unsupported shape)
      ├─ s.barrier ≫   → too many phases; collapse (Section 8.3)
      └─ healthy       → 4)
   │
   ▼
4) probe_targeted_bench against baseline (Triton / CK Tile / AITER)
      │
      ├─ regression    → 5)
      └─ near-best     → ship & document
   │
   ▼
5) probe_isa_inspect
      │
      ├─ buffer_store_short ≫ → epilogue is scalar; vectorize (Section 9.2)
      ├─ ds_read non-tr ≫     → swizzle isn't paired with reader; pick the right LDS layout
      ├─ waitcnt patterns ≫   → barrier/wait timing not aligned to async DMA
      └─ unclear              → 6)
   │
   ▼
6) rocprofv3 with metrics.txt
      │
      ├─ memory_stall > 40 %  → prefetch / async LDS (Section 8.2)
      ├─ lds_stall > 20 %     → run analyze_lds_conflicts.py
      ├─ bandwidth > 80 %     → memory-bound, increase reuse (Section 6.3)
      └─ compute_util > 70 %  → optimize MFMA packing (Section 7.4)
```

This decision tree is intentionally lossy: it points at the next
section to read, not at the final answer. The runbook itself is the
canonical reference; this tree is just the dispatcher.

### Symptom-to-action table (complementary view)

When you have a known symptom and want the typical first action:

```text
Symptom: correct but slow, low MFMA count
Likely:  wrong atom / tile, scalar path, missing vectorization
Action:  inspect LLVM / ISA (§11, §18); check atom selection (§7.1)
         and the inner loop (§12.1.B/C)

Symptom: fast but incorrect only on padded / tail shapes
Likely:  invalid pointer load, bad descriptor valid, vector crosses tail
Action:  test tiny adversarial shapes (§1.5); inspect buffer-rsrc
         sentinel path (§6.1, §21.6)

Symptom: intermittent wrong answers in async path
Likely:  missing `s_waitcnt` / barrier, workspace lifetime
Action:  add / check `s_waitcnt(vmcnt=0)` (§8.2-8.3), stream sync,
         launcher keep-alive (§12.1.N)

Symptom: atom change improves ISA but regresses runtime
Likely:  VGPR / LDS occupancy loss, or epilogue bottleneck
Action:  inspect resources via `probe_occupancy.py`; try cshuffle vs
         direct epilogue alternatives (§9.2-9.3)

Symptom: direct conv close but not within tolerance
Likely:  K-packed lane order or accumulator-reset bug
Action:  compare per-lane small reference; inspect fold order
         (§5.3, §7.3)

Symptom: low MfmaUtil + high VALUBusy + low MemUnitStalled
Likely:  compute-throttled by VALU / LDS plumbing, not HBM-bound
Action:  per-iter ISA histogram (§17.4); fix the plumbing
         (epilogue / register-residency / lane layout), not the
         launch knobs

Symptom: removing a per-iter cost regressed throughput
Likely:  the cost moved to a worse place (cross-lane permute,
         VGPR pressure, occupancy drop)
Action:  per-iter ISA diff before/after the change (§9.3, §17.4)
         — never declare a structural change a win without it

---
Each node maps to a lever section in [20-levers/](../20-levers/) or to [dsl-probe-workflow](../10-diagnosis/dsl-probe-workflow.md). Heuristics behind it: [decision-heuristics](../40-failure-reporting/decision-heuristics.md).
