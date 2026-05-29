---
name: steps-01-04-baseline-loop
description: Steps 1-4: static occupancy probe, hygienic sweep (small-N tile wins, K-pack saves 25%), standalone bf16 correctness, and the honest 4.38x-slower-than-rocBLAS baseline.
source: ck-dsl-gemm-skinny-decode-README.md (lines 83-217)
---

## Step 1 — Static occupancy probe (`01_probe_occupancy.py`)

> Runbook §3.1b "Static Inspection First"; §12.1.Q probe catalog;
> §14.2 "Low occupancy" / "Register spills" failure modes.

Compile each candidate, parse VGPR / AGPR / SGPR / LDS / spill from the
HSACO via `llvm-readelf --notes`, estimate `waves/CU` and the dominant
limiter, and drop variants that spill or fall below 4 waves/CU.

Output excerpt (full table in `data/01_occupancy.json`):

```
label                              vgpr agpr sgpr spill  lds B   waves/CU  wg/CU  limit
t16x64x32_w1x2_a16x16x32_mem        56    8   20    0    7168       32     16  MAX_WAVES_PER_CU
t16x128x32_w1x2_a16x16x32_mem       84   16   20    0   13312       20     10  VGPR
t16x128x64_w1x2_a16x16x32_mem      104   16   20    0   22528       14      7  LDS
t16x256x32_w1x4_a16x16x32_mem       88   16   20    0   25600       20      5  VGPR
t32x128x32_w2x2_a16x16x32_mem       60   16   20    0   18432       32      8  MAX_WAVES_PER_CU
t16x128x32_w1x2_a16x16x32_compv3    84   16   20    0   13312       20     10  VGPR
t16x128x32_w1x2_a16x16x32_compv4    84   16   20    0   13312       20     10  VGPR
t16x128x64_w1x2_a16x16x32_compv4   110   22   20    0   22528       14      7  LDS
```

All 8 variants survive the filter (zero spill, ≥4 waves/CU on every one).
The probe's diagnostic value is still real:

- The `compv4` row paid +6 AGPRs and +0 VGPRs over `mem` at the same tile.
  That accounting is invisible from latency alone — and is the kind of
  pre-bench signal §12.1.D recommends checking before deciding pipeline.
- `t16x128x64` (wider K window) dropped `waves/CU` from 20 to 14 by hitting
  the LDS limit. Predicts a potential occupancy regression from the K-window
  knob even before measurement.

## Step 2 — Sweep with hygiene (`02_sweep_bench.py`)

> Runbook §2.2 / §2.3 baseline + hygiene; §12.1.P knob list;
> §12.2 sweep discipline.

For each surviving variant: compile once (HSACO cached on disk), then
**5 timed attempts + 1 cold attempt** through `ck_dsl.run_manifest` with
**20 warmup + 200 timed** iterations per attempt. Report median, spread,
best. Salt the kernel symbol with the shape hash (`__m{M}n{N}k{K}` suffix).

Result on this hardware (full data in `data/02_sweep_bench.json`):

| Variant | Median ms | Spread | Best ms | TFLOPS | % HBM |
|---|--:|--:|--:|--:|--:|
| **t16x64x32_w1x2_a16x16x32_mem**   | **0.0483** | 0.1 % | **0.0483** | **1.39** | **8.7 %** |
| t16x128x64_w1x2_a16x16x32_mem      | 0.0490 | 0.1 % | 0.0490 | 1.37 | 8.6 % |
| t16x128x64_w1x2_a16x16x32_compv4   | 0.0529 | 0.1 % | 0.0528 | 1.27 | 7.9 % |
| t16x128x32_w1x2_a16x16x32_mem      | 0.0652 | 0.1 % | 0.0651 | 1.03 | 6.4 % |
| t16x128x32_w1x2_a16x16x32_compv3   | 0.0651 | 0.0 % | 0.0651 | 1.03 | 6.4 % |
| t16x128x32_w1x2_a16x16x32_compv4   | 0.0651 | 0.1 % | 0.0651 | 1.03 | 6.4 % |
| t32x128x32_w2x2_a16x16x32_mem      | 0.0693 | 0.3 % | 0.0691 | 0.97 | 6.1 % |
| t16x256x32_w1x4_a16x16x32_mem      | 0.0793 | 0.5 % | 0.0791 | 0.85 | 5.3 % |

What the data says — read it against the runbook:

- **The smallest-N tile wins.** For `M=2` the kernel is HBM-bound; covering
  fewer weight columns per CTA means more CTAs cover more of the N axis in
  parallel, and `MAX_WAVES_PER_CU` is the limiter (probe step). §3.3
  "Memory-bound signals".
- **`compv4` did not help on the matching tile.** `t16x128x32` is identical
  to within noise across `mem` / `compv3` / `compv4` (0.0651 ms × 3). The
  runbook §12.1.D notes "Compv3 / compv4 trade LDS for latency hiding";
  here, with no compute to hide behind, the trade is neutral. The §17.4
  "compiler hint sweep" lesson generalizes: schedule knobs rarely close
  memory-bound gaps.
- **The K-window hypothesis** (`t16x128x64`) saved ≈25 % of latency vs
  `t16x128x32` by halving the K-loop trip count via the same `16x16x32`
  atom — the §7.4 / §12.1.C "K-pack" lever, confirmed.
- **Wider N (`t16x256x32`) regressed by 64 %.** Confirms that for `M=2` the
  reuse story doesn't pay; you're paying VGPR / LDS for an N slab no lane
  fully consumes. §6.5 register pressure.
- **Best DSL kernel: `t16x64x32_w1x2_a16x16x32_mem` — 48.3 µs / 1.39 TFLOPS /
  695 GB/s / 8.7 % of HBM peak.**

## Step 3 — Standalone bf16 correctness (`03_correctness.py`)

> Runbook §2.1; §14.1 "if correctness fails, do not report speed as a win";
> §14.3 "Verification included in timing" caveat.

We do **not** use `ck_dsl.run_manifest --verify` for `bf16` because the
shipped verify path allocates `np.float16` buffers regardless of dtype, and
the bit pattern of fp16 reinterpreted as bf16 is garbage. (Concretely:
`max_abs_diff` of `2719.59` with `bad=8192/8192` on a perfectly correct
kernel — a structural false alarm in the harness, not a kernel bug.)

The replacement is ~80 lines: numpy `int16 → fp32 → bf16` for A and B, fp32
reference, launch the winning HSACO via `ck_dsl.runtime.hip_module.Runtime`,
read back, compare.

Result:
```
Verifying winner: t16x64x32_w1x2_a16x16x32_mem
  max|out-ref|=0.0000   bad=0/8192   ref_max=2736
  → PASS
```

Bit-exact. Time-to-execute the correctness check: ~5 s.

## Step 4 — Side-by-side vs rocBLAS (`04_compare_rocblas.py`)

> Runbook §2.2 same-harness comparison; §17 case-study reporting form.

Run `torch.matmul(A_bf16, W_bf16.T)` on the same shape with the same warmup
/ iter budget. Time with `torch.cuda.Event`.

| | Best ms | TFLOPS | GB/s | % HBM |
|---|--:|--:|--:|--:|
| **rocBLAS bf16 (torch.matmul)** | 0.0110 | 6.09 | 3047 | **38.1 %** |
| **DSL winner** (`t16x64x32`)    | 0.0483 | 1.39 |  695 |   8.7 % |
| **Ratio**                       | 4.38×  | —    | —    | —     |

Honest read:

- **DSL is 4.38× slower than rocBLAS** on this shape.
- The rocBLAS `38.1 %` here is higher than the `29 %` measured against ATOM
  in Addendum B. The two numbers are not the same experiment: the standalone
  bench has hot L2 (W reused across 1000+ iters), while the ATOM bench
  measures `o_proj` once per decode step against a freshly evicted
  weight. The original "29 %" finding is correct *for that workload*; the
  `38.1 %` here is what the kernel does when given perfect cache state.
  Document both — don't pick the one that matches your thesis (§14.3
  "Cache-biased results").
- The runbook's "small tile / `mem` pipeline / `kpack` atom" picks moved
  the needle 25 % from the median sweep row (0.0651 → 0.0483 ms), but the
  remaining gap to rocBLAS is **structural**, not knob-flippable:
  - rocBLAS uses a dedicated `skinny_gemm` codepath
    (`aiter.tuned_gemm.solMap["skinny"]`) tuned for `M ≤ 8`. We are using
    the same `UniversalGemmSpec` template that wins prefill — wrong
    algorithmic family per §12.1.A ("small / decode shapes → `flatmm`").
  - The §17.4 take-away applies: *a structural change often needs multiple
    co-evolved levers*. Tile geometry alone, with the wrong loop body, is
    capped at ~9 % HBM regardless of which tile you pick.

---
Method: [runbook-discipline](../00-method/runbook-discipline.md). Next: [step-05-extra-levers](step-05-extra-levers.md).
