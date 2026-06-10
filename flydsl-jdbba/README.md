# FlyDSL jagged grouped-GEMM (jdbba) — idioms, optimization, methodology

Porting Meta's HSTU `jagged_dense_bmm_broadcast_add` (jdbba) to AMD **FlyDSL**
(Flexible Layout Python DSL) on **MI355X / CDNA4 (gfx950)**, and optimizing it to
parity-or-ahead of the upstream autotuning Triton kernel.

**Thesis:** at the production shapes this is a **memory-bandwidth-bound** grouped
GEMM whose binding traffic is per-group `Dense[b]` weight re-reads, so the levers
that win are the ones that **cut HBM traffic** (cross-block L2 reuse via a chiplet
XCD remap, epilogue store vectorization) — not compute scheduling — and the
hardest part is **measuring honestly** against an autotuning baseline.

## Source

Distilled from a multi-session FlyDSL kernel-authoring + optimization effort
(2026-06-06 .. 2026-06-10): `examples/05-jagged_dense_bmm.py` (prototype),
`aiter/ops/flydsl/kernels/jagged_dense_bmm_gen.py` (production, uniform path),
`jagged_dense_bmm_persist_dev.py` (persistent skew path),
`jagged_dense_bmm_dispatch_v2.py` (regime dispatch), and the repo-root reports
`jagged_dense_bmm_optimization_report.md` / `..._followup_experiments.md`.

> **Verify live paths before relying on them.** These notes cite FlyDSL/aiter
> source paths and symbols that drift; the installed venv can also lag the repo.
> Confirm a symbol still exists before acting on a recommendation.

## Reading order

### 00 · Layout-API idioms (the reusable how-to)
- [varlen-jagged-idioms](00-layout-api-idioms/varlen-jagged-idioms.md) — pure
  `flydsl.expr` idioms for varlen/jagged kernels: device scalar read, runtime
  base-offset views, runtime early-exit (`scf.IfOp` positive guard), **bounded
  buffer descriptors** (HW OOB-drop for partial tiles), plain-B MFMA K-layout,
  bf16 broadcast-bias epilogue, and the `scf.for`-persistent-loop gotchas.

### 10 · Optimization case study
- [01-problem-and-roofline](10-optimization-case-study/01-problem-and-roofline.md)
  — the HSTU `(B,D,K,N)` shape-naming trap, the 4 headline shapes, and the
  roofline analysis proving memory-bound → why the binding lever is cross-block
  L2 reuse, not compute.
- [02-winning-levers](10-optimization-case-study/02-winning-levers.md) — the five
  levers that worked: LDS C-shuffle epilogue (+6–17%), shape-dependent BLOCK_K,
  XCD chiplet remap (+5%), the 16x16x32 bf16 MFMA atom (+4–7%), the XCD-aware
  persistent visiting order for skew (+2–5%), plus the zero-host-sync persistent
  problem-visitor kernel.
- [03-dead-ends](10-optimization-case-study/03-dead-ends.md) — every verified
  negative (bigger tiles, STAGES_A=3, B-in-LDS, ping-pong, async+BLOCK_M, 32x32
  atoms, DirectToLDS, finer autotune…) and *why* each fails at the byte floor.

### 20 · Methodology
- [measurement-methodology](20-methodology/measurement-methodology.md) — the
  autotune-median trap (p10/min never median), cold-L2 vs hot-L2, why wall-clock
  lies at small shapes, device-time-only, cosine+mean-error correctness, and the
  change-one-thing / re-measure-surprises discipline.

## Related topics

- [hipkitten/](../hipkitten/) — the AMD CDNA chiplet (XCD) block-ID remap
  (Algorithm 1) used here as the headline L2-reuse lever, and tile-DSL background.
- [ck-dsl-runbook/](../ck-dsl-runbook/) — the general GPU-kernel optimization
  method (diagnose → lever → autotune → verify) this case study instantiates.
