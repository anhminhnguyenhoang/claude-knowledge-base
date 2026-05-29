---
name: steps-06-11-existing-levers
description: Steps 6-11: rocprof reveals hipBLASLt's real kernel is MT16x16x512 (not the invented YAML story); pushing tile_n=16/tile_k=512 + preshuffle_b + hipcc backend closes 4.38x to 1.28x.
source: ck-dsl-gemm-skinny-decode-README.md (lines 298-425)
---

## Steps 6–11 — closing the gap with what the DSL already has

The original step-04 conclusion ("4.38× slower, blocked on structural
extensions to `streamk_gemm`") was wrong in one important way: it never
asked what hipBLASLt's actual kernel looks like *for this shape*, only
what the YAML library suggested for similar shapes. Running `rocprof`
on `torch.matmul(A, W.t())` (M=2 N=4096 K=4096) gave the real Tensile
kernel name:

```
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs
  _MT16x16x512  _MI16x16x1  _DTLA1_DTLB1
  _GRVWA8_GRVWB8  _GSU0  _PGR2_PLR1  _SIA3
  _SK3_SKFTR0  _WG16_4_4
                                AverageNs=9076  (≈9 µs)
```

Translation:

| Tensile token | Meaning | DSL equivalent today |
|---|---|---|
| **MT 16×16×512** | macro tile (the whole story) | `tile_m=16, tile_n=16, tile_k=512` |
| MI 16×16×1 | MFMA atom | `mfma_f32_16x16x32_bf16` |
| GRVWA8 / GRVWB8 | vector-width 8 on both operands | DSL emits dwordx8 bursts already |
| **DTLA1 / DTLB1** | DirectToLDS for A and B | **not exposed in DSL today** |
| **SK3** | StreamK basic, no split-K (GSU0) | **not exposed in DSL today** |
| PGR2 / PLR1 | 2 prefetch-global, 1 prefetch-local | `mem` pipeline (implicit-1) / `compv4` (2) |
| SIA3 | ScheduleIterAlg=3 (fixed loop schedule) | implicit per pipeline |
| WG 16,4,4 | 256-thread workgroup, 4×4 wave grid | analogous to our `warp_m × warp_n` |

The earlier README invented a "MT 256×192, GSU=34" story from a YAML
that is tuned for a totally different M-range. The real kernel is
**MT 16×16×512** — the exact same tile family the DSL can express,
just with a much deeper `tile_k`. That changes the punch list from
"missing three structural extensions" to "push the levers we already
have and see how close we get".

### Step 6 — try the geometries hipBLASLt actually picks (`06_ground_truth_geometries.py`)

Ground-truth-inspired sweep: keep `mfma_f32_16x16x32_bf16`, sweep
`tile_n ∈ {16, 32}` and `tile_k ∈ {32, 64, 128, 256}` on
`tile_m=16 w1×{1,2,4}`. The smallest-N + deepest-K candidate the
runner accepts wins.

**Winner: `t16x16x256_w1x1_mem`  16.0 µs / 26.2 % HBM / 1.47× rocBLAS.**
A single lever — bumping `tile_n` from 64 to 16 and `tile_k` from 32 to
256 — closed **3.02×** of the 4.38× gap. Preshuffle-B variants in this
script all fell over (`run_manifest` doesn't pass through
`--preshuffle-b`); we fixed that in step 9.

### Step 7 — push `tile_k` further (`07_push_tile_k.py`)

LDS budget on `t16x16` with the `mem` pipeline is `4·tile_k + 512` bytes,
so the 160 KiB cap leaves us room up to `tile_k = 4096`. Sweep
`tile_k ∈ {256, 384, 512, 768, 1024, 1536, 2048}` × `{mem, compv4}`:

| tile_k | `mem` (µs) | `compv4` (µs) | note |
|---:|---:|---:|---|
| 256  | 16.0 | 16.4 | step-6 winner |
| 384  | —    | —    | divisibility fail, skipped |
| **512**  | **13.5** | 13.5 | sweet spot |
| 768  | —    | —    | divisibility fail |
| 1024 | 58.4 | 58.6 | occupancy collapse |
| 1536 | 58.7 | LDS-OOM | compv4 hits 160 KiB at 197 KB |
| 2048 | 58.6 | LDS-OOM | |

**Winner: `t16x16x512_mem`  13.5 µs / 31.0 % HBM / 1.23× rocBLAS.**
This is hipBLASLt's exact `MT 16×16×512`. The collapse at `tile_k ≥ 1024`
is the LDS pressure crowding waves below the round-trip-hiding threshold.

### Step 8 — combinatorial lever sweep (`08_lever_combinations.py`)

192 combos of `pipeline × scheduler × epilogue × persistent × chiplet ×
waves_per_eu` locked to `t16x16x512`. Best is
`mem_interwave_cshuffle_pers` at **13.46 µs**; top-10 cluster within
0.04 µs (noise). The runbook §17.4 lesson holds at this geometry too:
scheduling knobs don't close memory-bound gaps once geometry is right.

### Step 9 — preshuffle_b (`09_preshuffle_b.py`)

`TraitSpec.preshuffle_b=True` is fully wired into `emit_load_phase`
(`gemm_universal.py:702-740`) — it replaces strided B loads with
contiguous `buffer_load_dwordxN` bursts. `run_manifest` doesn't know
how to permute B, so this script builds its own harness around
`ck_dsl.runtime.hip_module.Runtime` and the host-side
`np.transpose(B.reshape(n_tiles, bn, k_tiles, bk), (2,0,1,3))`
permutation. Bit-exact correctness (max_abs_diff = 0).

**Winner: `t16x16x512_preB`  13.33 µs / 31.5 % HBM / 1.21× rocBLAS.**
~1.5 % over non-preshuffle — the load addressing was already close to
optimal on this skinny-M shape; preshuffle helps more when the strided
load pattern is the actual bottleneck.

### Step 10 — `lower_kernel_to_hip` + hipcc backend (`10_hipcc_backend.py`)

The DSL has a second compile path: `compile_kernel_via_hipcc` lowers
the same IR to HIP C++ and runs it through `hipcc --genco -O3`. Same
runtime ABI, different codegen frontend (clang for HIP vs the direct
LLVM-IR → libamd_comgr path). Tested 5 variants:

| backend / flags | µs | compile |
|---|---:|---:|
| llvm_default (libamd_comgr) | 13.42 | 62 ms |
| hipcc -O3 | 13.42 | 407 ms |
| hipcc -O3 -ffast-math | 13.36 | 400 ms |
| hipcc -O3 -ffast-math -fgpu-flush-denormals-to-zero | 13.41 | 400 ms |
| hipcc -O3 -ffast-math -flushdenorm -mllvm -amdgpu-early-inline-all=true | 13.40 | 404 ms |

All within 0.06 µs (noise). For a memory-bound kernel the codegen
backend doesn't matter; for a long-running attention kernel the
helpers/compile.py docstring's ~5 % hipcc win could plausibly show up,
just not here.

### Step 11 — final side-by-side (`11_final_compare.py`)

Same `torch.cuda.Event` harness as step 04, so the ratio is honestly
comparable to the original 4.38×:

| | Best µs | TFLOPS | GB/s | % HBM |
|---|--:|--:|--:|--:|
| **rocBLAS bf16 (torch.matmul)** | 10.4 | 6.48 | 3243 | **40.5 %** |
| **DSL `t16x16x512` (mem/interwave/cshuffle)** | 13.2 | 5.07 | 2537 | 31.7 % |
| **Ratio** | **1.28×** | — | — | — |

DSL vs rocBLAS `max_abs_diff = 0.5` (one ulp of bf16 — both kernels
round their output bf16, accumulation order differs).

**Closed from 4.38× → 1.28×. Speedup over the original step-02 winner: 3.65×.**

---
Prev: [step-05-extra-levers](step-05-extra-levers.md). Next: [steps-12-14-direct-to-lds](steps-12-14-direct-to-lds.md).
