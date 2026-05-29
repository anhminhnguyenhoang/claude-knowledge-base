---
name: runbook-discipline
description: The CK DSL Optimization Runbook loop (static-probe → sweep → correctness → side-by-side → levers → confirm), hardware/software pin, reproduce steps, and the per-script runbook-section index.
source: ck-dsl-gemm-skinny-decode-README.md (lines 22-81, 690-722)
---

## What this example shows

1. **The runbook's static-first discipline (§3.1b, §12.1.Q).** Eight tile /
   pipeline variants are filtered by VGPR / AGPR / SGPR / LDS / spill /
   waves-per-CU *before* any kernel is launched. Cost: ~6 s of compile, no GPU.
2. **The runbook's sweep discipline (§12.2, §12.1.P).** Each surviving
   variant is benchmarked with `--attempts 5`, cold-cache discarded,
   median + spread reported, and the HSACO symbol is salted with the shape
   hash so module-cache aliasing cannot poison the result.
3. **Correctness is verified separately (§2.1, §14.1).** `ck_dsl.run_manifest`
   `--verify` allocates `fp16` buffers regardless of manifest dtype, so it
   silently fails on `bf16`. We re-run the kernel in a small Python harness
   with actual `bf16` (via torch) and check `max_abs_diff` against an `fp32`
   reference.
4. **Side-by-side vs rocBLAS (§2.2, §17 case-study form).** Same harness,
   same warmup / iter budget, same shape. Report the ratio honestly.

## Hardware / software pin

| | |
|---|---|
| GPU | MI355X / gfx950 |
| ROCm | 7.0.2 |
| torch | 2.12.0+rocm7.2 |
| CK DSL | this repo (`projects/composablekernel/python/ck_dsl`) |
| HBM peak | 8.0 TB/s |
| BF16 MFMA peak | 2.5 PFLOPS |

## Reproduce end-to-end

```bash
cd <ck_dsl>/examples/gemm_perf_skinny_decode    # this folder
PY=<your python with torch+rocm and ck_dsl on PYTHONPATH>

$PY scripts/01_probe_occupancy.py     # ~6 s,  CPU only (static)
$PY scripts/02_sweep_bench.py          # ~3 min,  GPU
$PY scripts/03_correctness.py          # ~5 s,   GPU
$PY scripts/04_compare_rocblas.py      # ~5 s,   GPU
$PY scripts/05_extra_levers.py         # ~2 min,  GPU
$PY scripts/06_ground_truth_geometries.py   # ~5 min,  GPU
$PY scripts/07_push_tile_k.py               # ~3 min,  GPU
$PY scripts/08_lever_combinations.py        # ~25 min, GPU (192 combos)
$PY scripts/09_preshuffle_b.py              # ~1 min,  GPU
$PY scripts/10_hipcc_backend.py             # ~3 min,  GPU
$PY scripts/11_final_compare.py             # ~10 s,   GPU
$PY scripts/12_direct_to_lds.py             # ~30 s,   GPU
$PY scripts/13_dtl_sweep.py                 # ~2 min,  GPU — DTLA × tk × cache hints
$PY scripts/14_dtl_push.py                  # ~30 s,   GPU — push tk past 1024
$PY scripts/18_dtl_prefetch.py              # ~1 min,  GPU — ping-pong DTLA (perf-neutral)
$PY scripts/19_multiwarp_probe.py           # ~30 s,   GPU — isolate multi-warp bug
$PY scripts/20_dtl_multiwarp.py             # ~3 min,  GPU — multi-warp resweep
$PY scripts/21_chiplet.py                   # ~5 min,  GPU — chiplet_swizzle sweep (winner)
$PY scripts/22_confirm_winner.py            # ~5 min,  GPU — N=60 confirmation
```

> Steps 18–22 require the patched `ck_dsl/instances/gemm_universal.py`
> (direct `async_buffer_load_lds_addr` wiring + per-wave LDS base
> offset). Without those, multi-warp DTLA produces incorrect output.

Each step writes to `data/0N_*.json`. The next step reads from the previous one.

## Runbook section index (where each script's discipline came from)

| Script | Runbook anchors |
|---|---|
| `01_probe_occupancy.py` | §3.1b, §12.1.Q, §14.2 |
| `02_sweep_bench.py`     | §2.2, §2.3, §12.1.B–H, §12.1.P, §12.2 |
| `03_correctness.py`     | §2.1, §14.1, §14.3 |
| `04_compare_rocblas.py` | §2.2, §17 (case-study form), §14.3 |
| `05_extra_levers.py`    | §12.1.G, §12.1.I, §12.1.L, §17.4 (didn't-help table) |
| `06_ground_truth_geometries.py` | §12.1.B–C (tile/K-pack), §17.4 (rocprof reading) |
| `07_push_tile_k.py`     | §12.1.C (K-pack), §6.5 (LDS budget), §3.3 (memory-bound) |
| `08_lever_combinations.py` | §12.1.D–L (full knob matrix), §17.4 (compose-or-not) |
| `09_preshuffle_b.py`    | §12.1.J (preshuffle), §2.1 (custom-harness correctness) |
| `10_hipcc_backend.py`   | §12.1.R (codegen backend), §14.3 (noise threshold) |
| `11_final_compare.py`   | §2.2, §17, §15 (final form) |
| `12_direct_to_lds.py`   | §12.1.R (codegen surface extension), §17.4 (neutral/negative-result form) |
| `13_dtl_sweep.py`       | §12.1.D–L (knob composition), §12.1.R (cache hints) |
| `14_dtl_push.py`        | §6.5 (LDS budget), §3.1b (occupancy frontier) |
| `15_warp_geom.py` / `16_more_lanes.py` / `17_confirm_wpe.py` | §12.1.E (warp grid), §17.4 (negative-result form) — surfaced the multi-warp DTLA bug |
| `18_dtl_prefetch.py`    | §12.1.D (pipeline depth), §17.4 (compose-or-not, neutral result) |
| `19_multiwarp_probe.py` | §2.1, §14.1 (correctness as a probe), §17.4 (isolate before patching) |
| `20_dtl_multiwarp.py`   | §3.3 (memory-bound saturation), §12.1.E (warp grid) |
| `21_chiplet.py`         | §12.1.L (chiplet swizzle) — applied at the right shape this time |
| `22_confirm_winner.py`  | §2.3 (hygiene), §14.3 (paired-vs-standalone reference) |

## CK Tile examples that inspired this layout

| CK Tile path | What it gave us |
|---|---|
| `example/ck_tile/03_gemm/universal_gemm.cpp` | `persistent`, `chiplet_swizzle`, `waves_per_eu` trait surface |
| `example/ck_tile/03_gemm/gemm_splitk_two_stage*.cpp` | Two-stage split-K is the canonical skinny-M pattern (named as the unaddressed follow-up) |
| `example/ck_tile/18_flatmm/flatmm_basic.hpp` | `FlatmmConfig32` / `Config16` tile defaults — the "small-shape" templates the DSL `flatmm` instance mirrors |
| `example/ck_tile/40_streamk_gemm` | Where the next move would land (rectangular small-M tiles + bf16) |

---
Walkthrough begins: [steps-01-04-baseline-loop](../10-walkthrough/steps-01-04-baseline-loop.md). Outcome: [ceiling-and-followups](../20-conclusion/ceiling-and-followups.md).
