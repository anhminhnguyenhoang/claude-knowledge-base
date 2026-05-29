---
name: failure-modes
description: Runbook S14: correctness failures (lane mapping, stale LDS, REGS_PER_LANE invariants, validator-outlives-constraint, HSACO cache aliasing), performance failures (scalar stores, spills, low occupancy), and benchmark failures.
source: ck-dsl-optimization-runbook.md (lines 1824-1893)
---

## 14. Failure Modes

### 14.1 Correctness Failures

- Wrong lane mapping.
- Wrong vector element order.
- Missing boundary zero.
- Stale LDS data.
- Missing barrier.
- Wait without barrier for cross-thread data.
- Incorrect tail store.
- Wrong group / head offset.
- Wrong stride.
- Wrong K packing.
- Accumulator slot reset too early or too late.
- Compiler flag miscompile (Section 10.3).
- DSL AST rewriter elision (Section 10.3 — only relevant for
  tracing-style DSLs; CK DSL guards this via `Value.__bool__`).
- HIP-debug backend missing an op lowering (run
  `probe_lowering_compare.py` to detect).
- **`REGS_PER_LANE`-dependent invariant left over from a previous
  atom.** When switching MFMA shape (e.g. 16×16 → 32×32 quadruples
  `REGS_PER_LANE`), any code that writes only "slot 0" of a per-lane
  state tensor will silently cover 1/4 or 1/16 of the lanes. The
  sinks initialization bug in **§17.4** is the canonical example —
  it manifested as an elevated `max_abs_diff` localized to specific
  query positions while the rest of the output was bit-exact.
- **Spec-validator restriction outlives the constraint it protected.**
  A "to be safe" validator block that prevents two flags from
  combining can outlive the v1 restriction it documented; if the
  restriction is no longer real, the validator is silently blocking
  the winning code path. Audit each cross-flag restriction
  periodically.
- **HSACO module-cache aliasing across specializations.** When the
  display kernel symbol is the same across two compile-time-constant
  variants, the cache can serve the wrong HSACO. Salt the kernel
  symbol with a shape hash before compile (see §12.2).

### 14.2 Performance Failures

- Scalar stores (`probe_isa_inspect.py`: `buffer_store_short` >> 0).
- Scalar loads.
- Excessive barriers.
- Excessive waits.
- Register spills (`probe_occupancy.py`: `spill > 0`).
- Low occupancy (`probe_occupancy.py`: `waves_per_cu < 4`).
- Bank conflicts (`analyze_lds_conflicts.py`).
- Bad grid scheduling.
- Tiny per-block work.
- Too much LDS.
- Over-general runtime index math.
- Uncoalesced metadata loads.
- Over-fused epilogue.

### 14.3 Benchmark Failures

- Measuring allocation.
- Measuring initialization.
- Missing synchronization.
- Not enough iterations.
- Thermal throttling (lock with `rocm-smi --setperflevel high`).
- Clock changes.
- Cache-biased results.
- Comparing different shapes / layouts.
- Comparing different precision.
- Verification included in timing.
- First-run JIT compilation included.
- `time_launches` vs Triton autotuner interaction — use
  `probe_targeted_bench.py::time_cuda_event` instead when timing
  Triton in the same window.

---
Heuristics for fixing each: [decision-heuristics](decision-heuristics.md). Bug signatures in practice: [unified-attention-2d](../05-case-studies/unified-attention-2d.md) (S17.4). Tolerance signatures: [define-the-problem](../01-diagnosis/define-the-problem.md) (S1.4).
