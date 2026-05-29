---
name: the-loop-and-how-to-read
description: The 10-step optimization loop (hypothesis -> correctness -> measure -> inspect IR/ISA -> one lever -> re-verify -> explain -> keep/revert -> record), and the document map. Read first.
source: ck-dsl-optimization-runbook.md (lines 1-92)
---

# CK DSL Optimization Runbook

This runbook is a long-form checklist for optimizing GPU kernels written
with `ck_dsl`.

Every section ties the general optimization concept to a concrete
`ck_dsl` primitive, helper, instance, or probe. The goal is that an
engineer with the codebase open can find the lever, change it, verify
it, measure it, and explain it — without leaving this tree.

Use it as a menu of considerations. Do not apply every item blindly.
Start from the operation contract and the measured bottleneck.

## The Loop (one-page summary)

If you take only one thing from this runbook, take this loop:

```text
1. State the hypothesis.
2. Verify correctness baseline.
3. Measure baseline with stable timing.
4. Inspect generated IR / ISA / resources (probes in §18).
5. Change one lever (catalog in §12.1).
6. Re-verify correctness.
7. Re-measure with the same harness.
8. Explain why it moved (per-iter ISA diff).
9. Keep or revert.
10. Record the result.
```

Do not batch several levers unless you are explicitly doing a coarse
search and plan to isolate the winner afterward. If stuck and no clear winning change with simple levers, extend to multiple levers in smart combinations and use judgement and do not give up and revert losses easily. **If correctness
fails, do not report speed as a win.**

## How to read this document

The structure mirrors the general runbook:

1. Define the problem (contract, shapes, layouts, dtypes, boundaries).
2. Establish baselines (correctness, performance, hygiene).
3. Classify the bottleneck (arithmetic intensity, profiler, IR/ISA).
4. Choose the algorithmic mapping.
5. Decompose the work (grid, block, wave, thread).
6. Optimize the memory hierarchy (global, LDS, registers, caches).
7. Pick matrix instructions and operand layouts.
8. Pipeline and schedule.
9. Optimize the epilogue.
10. Tune the compiler.
11. Inspect ISA and resources.
12. Autotune.
13. Apply operation-specific checklists (GEMM / conv / attention / …).
14. Recognize failure modes.
15. Report results.
16. Apply decision heuristics.
17. Read the empirical case studies.

Two extra sections are DSL-specific:

- **§18 DSL Probe Workflow** — when and how to use the probes under
  `utilities/tools/dsl_probes/`.
- **§19 Reproducible Commands** — exact venv / PYTHONPATH / harness
  invocations for the production workflows.

Cross references:

- `runbook_mapping.md` — section-by-section DSL-primitive table.
- `runbook_compliance.md` — empirical pass results per section.
- `measured_results.md` — last documented validation pass numbers.

If you came here looking for a specific knob, jump to **§12.1 Knob
Catalog** — it enumerates every performance lever exposed by `ck_dsl`,
grouped by family (algorithmic variant, tile geometry, MFMA atom,
pipeline, epilogue, LDS layout, occupancy, preshuffle, persistent /
Stream-K, quantization, attention-2D micro-levers, chiplet swizzle,
compiler flags, runtime / launch, dispatcher policy, benchmark
hygiene, static probes).

If you came here looking for **what is available on gfx950 / CDNA4
specifically** (the DSL's default target — MI350X / MI355X), jump to
**§21 Target Architecture Reference**: the full MFMA atom catalog,
LDS specs and bank-conflict rules, CDNA4-only intrinsics
(`v_permlane32_swap_b32`, `ds_read_tr16_b{64,128}`, `ds_read_tr_b8`,
scaled / MX MFMA), VGPR / AGPR / occupancy caps, chiplet swizzle
parameters, buffer-descriptor flags, fp8 / quantization support, and
gfx950 compiler caveats.
- `utilities/skills/` — focused skill docs (`gemm-optimization`,
  `lds-optimization`, `kernel-trace-analysis`,
  `prefetch-data-load`, `capture-kernel-trace`, `empirical-case-studies`,
  `kernel-launch-guide`, `bisect-perf-regression`).
- `utilities/tools/dsl_probes/` — the static-inspection probes
  introduced for this runbook.


---
Next: [define-the-problem](../01-diagnosis/define-the-problem.md) · jump to [knob-catalog-and-sweep](../03-autotuning/knob-catalog-and-sweep.md) for the lever menu · [reproducible-commands](reproducible-commands.md). For a full worked walkthrough applying this method end-to-end, see the [skinny-decode-gemm](../05-case-studies/skinny-decode-gemm/README.md) case study.
