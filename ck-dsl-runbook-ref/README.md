# CK DSL Optimization Runbook — reference

The **canonical CK (Composable Kernel) DSL Optimization Runbook**: a
long-form checklist + lever reference for optimizing GPU kernels written
with `ck_dsl`. Every general optimization concept is tied to a concrete
`ck_dsl` primitive, helper, instance, or static probe, so an engineer
with the codebase open can find the lever, change it, verify it, measure
it, and explain it.

This topic is the **menu of levers and the method**. For one problem
taken from 4.38× slower to parity, the
[skinny-decode-gemm](05-case-studies/skinny-decode-gemm/README.md)
case study is a full 22-step *worked walkthrough* of applying this
runbook end-to-end (the runbook's own condensed §17.7 summary of the
same pass lives beside it as
[runbook-17-7-condensed](05-case-studies/skinny-decode-gemm/runbook-17-7-condensed.md)).

## Source

Derived verbatim from the upstream document
`projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md`
on the ROCm `rocm-libraries` fork branch
`users/vanantha/ck-dsl-prototype`. The full document (3,133 lines) is
preserved in
[`../_sources/ck-dsl-optimization-runbook.md`](../_sources/ck-dsl-optimization-runbook.md)
for provenance.

> **Verify live paths before relying on them.** Most notes cite live
> source paths (`instances/*.py`, `helpers/*.py`,
> `utilities/tools/dsl_probes/*`) and line numbers in the upstream
> `ck_dsl` tree. These drift — confirm the
> [upstream path](https://github.com/ROCm/rocm-libraries/blob/users/vanantha/ck-dsl-prototype/projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
> still resolves and the referenced symbols still exist before acting on
> a recommendation.

## Reading order

### 00 · Method (read first)
- [the-loop-and-how-to-read](00-method/the-loop-and-how-to-read.md) — the 10-step optimization loop and the document map. **Start here.**
- [reproducible-commands](00-method/reproducible-commands.md) — exact venv / PYTHONPATH / harness / rocprof invocations (§19).

### 01 · Diagnose before you change
- [define-the-problem](01-diagnosis/define-the-problem.md) — operation contract, shapes, layouts, dtypes/tolerances, boundaries (§1).
- [establish-baselines](01-diagnosis/establish-baselines.md) — correctness gates, perf baselines, benchmark hygiene, metadata (§2).
- [bottleneck-classification](01-diagnosis/bottleneck-classification.md) — arithmetic intensity, rocprof PMC decision tree, static-probe tier, bottleneck signal lists (§3).
- [dsl-probe-workflow](01-diagnosis/dsl-probe-workflow.md) — the static-probe sequence (occupancy → intrinsic → ISA → bench → rocprof) and probe output fields (§18).

### 02 · The levers (lever family by lever family)
- [algorithmic-mapping](02-levers/algorithmic-mapping.md) — pick the kernel first; GEMM/conv/attention/reduction/transpose family tables + the implicit-GEMM VGPR tax (§4).
- [work-decomposition](02-levers/work-decomposition.md) — grid/block/wave/thread mapping; MFMA lane mapping; the K-pack lane-mapping bug (§5).
- [memory-hierarchy](02-levers/memory-hierarchy.md) — global loads/stores, LDS/async DRAM→LDS, bank conflicts (XOR vs padding per arch), registers, caches (§6).
- [matrix-instructions](02-levers/matrix-instructions.md) — MFMA/WMMA atom selection, operand layout, lane-layout matching across chained atoms, K-pack (§7).
- [pipelining-scheduling](02-levers/pipelining-scheduling.md) — software pipeline, async copy, waits/barriers, scheduling hints, instruction balance (§8).
- [epilogue](02-levers/epilogue.md) — direct vs LDS/CShuffle epilogue, the "in-between LDS epilogue loses both ways" caveat, output validation (§9).
- [compiler-build](02-levers/compiler-build.md) — build type, AMD/HIP flags, DSL-specific compiler hazards, alias semantics (§10).
- [isa-resource-inspection](02-levers/isa-resource-inspection.md) — the `ck_dsl.analysis` layer; what to count/extract/interpret (§11).

### 03 · Autotuning
- [knob-catalog-and-sweep](03-autotuning/knob-catalog-and-sweep.md) — **the master knob catalog** (§12.1.A–Q: every `ck_dsl` perf lever), sweep discipline, the dispatcher as a lever, variant naming (§12).
- [op-specific-checklists](03-autotuning/op-specific-checklists.md) — per-op tuning checklists: GEMM, conv, attention, reduction, fused (§13).

### 04 · Failure modes & reporting
- [failure-modes](04-failure-reporting/failure-modes.md) — correctness / performance / benchmark failure catalogs (§14).
- [reporting-template](04-failure-reporting/reporting-template.md) — experiment log, summary template, done-criteria (§15).
- [decision-heuristics](04-failure-reporting/decision-heuristics.md) — lever-direction heuristics + the anti-pattern list (§16).

### 05 · Empirical case studies
- [case-studies-overview](05-case-studies/case-studies-overview.md) — bake-off summary, validation pass, attention parity (the order-1 = structural-bug signature) (§17.0–17.3).
- [unified-attention-2d](05-case-studies/unified-attention-2d.md) — the headline worked example: closing a Triton gap via structural levers; 8 transferable principles (§17.4).
- [fused-moe](05-case-studies/fused-moe.md) — active-tile dispatch + preshuffle-B, two stacking MoE levers vs CK Tile C++ (§17.5–17.6).
- [skinny-decode-gemm/](05-case-studies/skinny-decode-gemm/README.md) — o_proj M=2 to 1.02× rocBLAS: a full 22-step worked walkthrough (its own sub-topic), plus the runbook's condensed §17.7 form beside it.

### 06 · Reference
- [target-architecture-gfx950](06-reference/target-architecture-gfx950.md) — gfx950/CDNA4 MFMA atoms, LDS specs, cross-lane primitives, register/occupancy caps, chiplet, fp8/MX, compiler caveats (§21).
- [diagnostic-decision-tree](06-reference/diagnostic-decision-tree.md) — the one-page decision tree + symptom-to-action table (Appendix).
- [cross-references](06-reference/cross-references.md) — pointers to the sibling `ck_dsl` docs, skills, probes, tool stages (§20).

### Glossary
This document defines its terms inline. For the Tensile kernel-name
tokens and CK DSL `TraitSpec` knobs decoded as a standalone table, see
the skinny-decode case study's
[glossary](05-case-studies/skinny-decode-gemm/glossary.md).

## One-sentence thesis

> Optimization is a disciplined loop, not a bag of tricks: classify the
> bottleneck with a cheap static probe before paying for rocprof, change
> **one lever** from a catalogued menu, confirm the change in the
> per-iter ISA histogram (not just latency), and remember that the cost
> you remove rarely disappears — it usually moves.
