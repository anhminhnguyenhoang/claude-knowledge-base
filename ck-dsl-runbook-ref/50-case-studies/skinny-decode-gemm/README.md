# Skinny-decode GEMM walkthrough (o_proj M=2)

> Case study under [ck-dsl-runbook-ref](../../README.md). This is the
> **full 22-step worked walkthrough**; for the runbook's own condensed
> §17.7 summary of the same pass see
> [runbook-17-7-condensed](runbook-17-7-condensed.md).

Notes on the **Composable Kernel (CK) DSL Optimization Runbook** applied
end-to-end to one real problem: a Qwen3-8B **`o_proj` decode** matmul
(`bf16`, `M=2`, `N=4096`, `K=4096`) — the worst-utilization GEMM in the
Qwen3-8B decode trace. Over 22 disciplined steps the DSL kernel closed from
**4.38× slower than rocBLAS** (48.3 µs) to **1.02× — within noise** (10.29 µs),
matching hipBLASLt's own `MT16x16x512` geometry on an MI355X (gfx950/CDNA4).

The value is the *loop*, not the number: static probe → hygienic sweep →
correctness → side-by-side → structural levers → high-confidence
confirmation. 14 of 22 steps were regressions or ties; the wins came from
three structural changes (DirectToLDS wiring, a per-wave LDS-offset fix that
made multi-warp DTLA correct, and chiplet swizzling), each surfaced only
because the sweep followed the rules.

## Source

Derived verbatim from the upstream example
`projects/composablekernel/python/ck_dsl/examples/gemm_perf_skinny_decode/README.md`
on the ROCm `rocm-libraries` fork branch `users/vanantha/ck-dsl-prototype`.
The README is preserved in [`../../../_sources/`](../../../_sources/) for provenance.

> **Artifacts are link-only.** The 22 step-scripts (`scripts/0N_*.py`) and 22
> benchmark results (`data/0N_*.json`) were *not* vendored. They live on a
> personal fork branch that may be force-pushed or deleted — verify the
> [upstream path](https://github.com/ROCm/rocm-libraries/tree/users/vanantha/ck-dsl-prototype/projects/composablekernel/python/ck_dsl/examples/gemm_perf_skinny_decode)
> still resolves before relying on it. Script/data filenames and the per-step
> runbook anchors are tabulated in [runbook-discipline](00-method/runbook-discipline.md).

## Reading order

### 00 · Method (read first)
- [runbook-discipline](00-method/runbook-discipline.md) — what the runbook loop is, the hardware/software pin, the reproduce sequence, and the per-script runbook-section index. **Start here.**

### 10 · The walkthrough (in order)
- [steps-01-04-baseline-loop](10-walkthrough/steps-01-04-baseline-loop.md) — static probe, hygienic sweep (small-N tile + K-pack win), bf16 correctness, and the honest 4.38× baseline.
- [step-05-extra-levers](10-walkthrough/step-05-extra-levers.md) — TraitSpec scheduling knobs all net neutral-or-loss on a memory-bound shape; split-K named as the real (unaddressed) structural lever.
- [steps-06-11-existing-levers](10-walkthrough/steps-06-11-existing-levers.md) — rocprof exposes hipBLASLt's actual `MT16x16x512`; pushing tile geometry + preshuffle + hipcc backend gets to 1.28×.
- [steps-12-14-direct-to-lds](10-walkthrough/steps-12-14-direct-to-lds.md) — wiring DirectToLDS; why it regresses 8% alone yet unlocks `tile_k=1024` to reach 1.01×.
- [steps-15-22-multiwarp-chiplet](10-walkthrough/steps-15-22-multiwarp-chiplet.md) — the multi-warp DTLA collision bug + per-wave LDS-offset fix, perf-neutral prefetch, and the final 2.6% chiplet-swizzle lift.

### 20 · Conclusion
- [ceiling-and-followups](20-conclusion/ceiling-and-followups.md) — the HBM-ceiling argument, the file map, and the runbook §15 final-summary template.

### Reference
- [glossary](glossary.md) — Tensile kernel-name tokens and CK DSL knobs decoded.

## One-sentence thesis

> On a memory-bound skinny-M decode GEMM, schedule knobs are noise and only
> three co-evolved *structural* levers — DirectToLDS, a per-wave LDS offset,
> and chiplet swizzling — close the gap to rocBLAS, which itself just sits at
> the part's ~44% sustained-bf16 HBM ceiling; beating it needs fewer bytes
> (fusion / quantization), not better GEMM tuning.
