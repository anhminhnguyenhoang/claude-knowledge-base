---
name: step-05-extra-levers
description: Step 5: CK-Tile-inspired TraitSpec knobs (waves_per_eu, persistent, chiplet_swizzle) all neutral-or-loss on a memory-bound shape; why split-K is the real structural lever; and the scoped-out non-goals (flatmm, split-K, FP8, ATOM swap).
source: ck-dsl-gemm-skinny-decode-README.md (lines 219-296)
---

## Step 5 — CK-Tile-inspired extra levers (`05_extra_levers.py`)

> Inspired by `example/ck_tile/03_gemm/universal_gemm.cpp` and
> `example/ck_tile/18_flatmm/flatmm_basic.hpp` (persistent CTAs,
> k_batch, chiplet-aware tile partitioner).
>
> Runbook §12.1.G `waves_per_eu`, §12.1.I `persistent`, §12.1.L
> chiplet swizzle, §17.4 "co-evolve multiple levers, record what didn't
> help".

Lock the geometry to the sweep winner and exercise the remaining
`TraitSpec` knobs. Result (`data/05_extra_levers.json`):

| Variant | Δms | Δ% | Verdict |
|---|--:|--:|---|
| baseline_winner | +0.0000 | +0.0 % | reference |
| waves_per_eu_2 | +0.0015 | +3.0 % | **loss** |
| waves_per_eu_3 | +0.0015 | +3.0 % | **loss** |
| persistent | -0.0000 | -0.0 % | noise |
| chiplet_swizzle | +0.0005 | +1.1 % | loss |
| persistent_chiplet | +0.0006 | +1.2 % | loss |
| compv4_persistent | -0.0000 | -0.0 % | noise |
| compv4_persistent_chiplet | +0.0006 | +1.2 % | loss |

Diagnostic read (§17.4 form):

- **`waves_per_eu`: regression.** The occupancy probe already showed
  `MAX_WAVES_PER_CU=32` is the limiter; forcing the compiler to a smaller
  VGPR budget can't add more waves but does shorten the live-range
  optimizer's freedom. §17.4: "a compiler hint that crosses no real
  constraint is at best neutral and usually a small loss".
- **`persistent`: noise.** The shape has only `M_tiles × N_tiles = 1 × 64 = 64`
  macro tiles. The non-persistent grid is already small; persistent's
  amortization of launch cost has nothing to amortize.
- **`chiplet_swizzle`: small loss.** MI355X has 8 XCDs; at 64 tiles
  the swizzle remaps `tiles ÷ chunk_size = 64 / 64 = 1` chunk, so the
  remap is a no-op for L2-reuse purposes and the extra entry-time
  scalar math is pure overhead. §12.1.L's "MI300X / MI325X / MI350X
  have 8 XCDs" works for shapes with many more tiles per N stripe.
- **Pipeline+persistent co-evolution: noise.** The §17.4 lesson "a
  structural change often needs multiple co-evolved levers" applies to
  *structural* changes (atom shape, lane ownership, intermediate
  residency). Stacking *scheduling-tier* knobs on a memory-bound kernel
  doesn't compose into a structural win.

The structural change that would actually move this kernel — and the
one CK Tile ships in `example/ck_tile/03_gemm/gemm_splitk_two_stage.cpp`
— is **split-K**: tile K into `k_batch` slabs, launch `k_batch ×` more
CTAs, reduce in a second-stage kernel. At `M=2, N=4096, K=4096` the
kernel processes 32 MiB of B weights against only 8 KiB of A; split-K
would parallelize the same B traffic across more CTAs and let many CUs
share the work currently confined to ~64 CTAs. That is the right
follow-up, and it requires *kernel-body* support that
`UniversalGemmSpec` does not expose today (`streamk_gemm` v1 ships f16
and uses `atomic_add` reduction, but its `tile_m` enum supports only
16/32 squares — not the rectangular small-M tiles this shape needs).

## What this example *doesn't* do (and why)

- **No `flatmm` variant.** `instances/flatmm.py` documents itself as the
  runbook's "small / decode" pick, but its v1 kernel body aliases
  `batched_gemm` and the preshuffled-B path is gated off — it won't differ
  from what we already swept. The honest move is a per-shape `skinny`
  variant in `gemm_universal` with a different load pattern; out of scope.
- **No split-K / two-stage reduce.** CK Tile's
  `gemm_splitk_two_stage.cpp` shows the canonical pattern: stage 1 writes
  `[k_batch, M, N]` partials, stage 2 reduces over `k_batch`. The DSL's
  `streamk_gemm` v1 has the partitioner + `Atomic` reduction in place but
  the spec is f16-only and the tile-shape enum doesn't accept the
  rectangular small-M tiles this shape needs. Bf16 support + rectangular
  tiles is the structural follow-up that would actually move the gap.
- **No FP8 KV / weight quantization.** §17.4 demoted FP8 KV with
  "`MemUnitStalled` near zero → not HBM bound, dequant adds VALU".
  Our PMC equivalent here would be ROCm Profiler `MemUnitStalled` on the
  measured kernel — adding it is a fair follow-up.
- **No ATOM monkey-patch.** Addendum C of the parent report covered the
  integration scaffolding; with this kernel measurably slower than rocBLAS,
  flipping the swap on inside ATOM would be a measured regression.

---
Prev: [steps-01-04-baseline-loop](steps-01-04-baseline-loop.md). Next: [steps-06-11-existing-levers](steps-06-11-existing-levers.md).
