---
name: ceiling-and-followups
description: The HBM ceiling argument (both kernels live at ~44% sustained-bf16 peak; beating rocBLAS needs fewer bytes via fusion/quantization, not GEMM tuning), the script/data file map, and the runbook §15 final-summary template.
source: ck-dsl-gemm-skinny-decode-README.md (lines 628-689, 724-757)
---

## Why this is the ceiling (and what would break it)

The shape transfers ≈33.5 MB total (B = 32 MiB, C = 16 KiB, A = 16 KiB).
At MI355X's 8 TB/s theoretical HBM peak, the *floor* is 4.2 µs. The
sustained streaming-bf16 ceiling on this part is ~3.5 TB/s ≈ 44 % of
peak — which is where both DSL (44 %) and rocBLAS (42 %) live.

To exceed rocBLAS on this shape requires **reducing total HBM bytes**,
not improving per-byte efficiency:

- **Kernel fusion** (the realistic win): fuse `o_proj` into the
  attention output reduction so the C tensor never round-trips HBM,
  and/or fuse the residual-add that follows. This drops C and the
  add's read from the byte total.
- **Weight quantization** (FP8 → halve B's 32 MiB). Outside the scope
  of "match rocBLAS bf16 → bf16".
- **Split-K** doesn't help here — A is already L1-resident across CTAs,
  B has no cross-CTA reuse, and both kernels saturate HBM. Splitting K
  would replicate B traffic.

The runbook loop converged on this kernel; the remaining gap is
algorithmic, not knob-tunable.

## File map

```
ck_dsl/examples/gemm_perf_skinny_decode/
├── README.md                              # this file
├── scripts/
│   ├── 01_probe_occupancy.py              # static probe, no GPU
│   ├── 02_sweep_bench.py                  # GPU sweep with hygiene
│   ├── 03_correctness.py                  # standalone bf16 verify
│   ├── 04_compare_rocblas.py              # side-by-side vs rocBLAS (original)
│   ├── 05_extra_levers.py                 # CK-Tile-inspired extras
│   ├── 06_ground_truth_geometries.py      # rocprof-derived MT16x16x* sweep
│   ├── 07_push_tile_k.py                  # push tile_k → MT16x16x512 winner
│   ├── 08_lever_combinations.py           # 192 scheduling combos on winner
│   ├── 09_preshuffle_b.py                 # preshuffle_b with custom harness
│   ├── 10_hipcc_backend.py                # LLVM vs hipcc-O3 backend
│   ├── 11_final_compare.py                # final apples-to-apples ratio
│   ├── 12_direct_to_lds.py                # DTLA/DTLB lever (wires upstream patch)
│   ├── 13_dtl_sweep.py                    # DTLA × tile_k × pipeline × cache-hints
│   ├── 14_dtl_push.py                     # push tile_k > 1024 (occupancy frontier)
│   ├── 15_warp_geom.py                    # multi-warp sweep (exposed DTLA correctness bug)
│   ├── 16_more_lanes.py                   # widen tile_n for multi-warp (garbage pre-patch)
│   ├── 17_confirm_wpe.py                  # waves_per_eu retest under multi-warp
│   ├── 18_dtl_prefetch.py                 # ping-pong double-buffered DTLA (bit-exact, perf-neutral)
│   ├── 19_multiwarp_probe.py              # 2×2 isolation of the multi-warp DTLA bug
│   ├── 20_dtl_multiwarp.py                # resweep wider tiles post-patch (slower, grid-bound)
│   ├── 21_chiplet.py                      # chiplet_swizzle sweep — 2.6% win
│   └── 22_confirm_winner.py               # N=60 paired confirmation of the chiplet winner
├── data/
│   ├── 01_occupancy.json … 11_final_compare.json
├── build/                                 # 02 sweep HSACOs
├── build_extra/                           # 05 extra-lever HSACOs
├── build_gt/                              # 06 ground-truth HSACOs
├── build_push_tk/                         # 07 tile_k sweep HSACOs
├── build_combo/                           # 08 lever-combo HSACOs
├── build_preB/                            # 09 preshuffle HSACOs
└── build_final/                           # 11 final-comparison HSACO
```


## Final summary (runbook §15 template)

```text
Best correct variant (step 21, confirmed step 22 N=60):
  name:        t16x16x1024  mem / interwave / cshuffle  +  DTLA + DTLB (CACHE_ALL/CACHE_ALL)
                                                        +  chiplet_swizzle wgm=8 chunk=16
  shape:       bf16 M=2 N=4096 K=4096 (Qwen3-8B o_proj decode)
  latency:     10.29 µs (best of 60, spread ~1%)
  throughput:  6.51 TFLOPS, 3263 GB/s (44% HBM)
  correctness: max_abs_diff vs fp32 reference = 0.0 (bit-exact)
  vs rocBLAS:  1.02× — within noise of rocBLAS (10.10 µs / 42% HBM, standalone)
  speedup vs original (step 02): 4.69×  (48.3 → 10.29 µs)
  key levers:  (1) DirectToLDS A & B in gemm_universal.py (TraitSpec.direct_to_lds)
                   removes the global→VGPR→ds_write round-trip (-32 VGPR, -32 inst/K-tile);
               (2) CACHE_ALL for BOTH operands (CACHE_STREAM/NT regress);
               (3) tile_k=1024 — only viable with DTLA (the round-trip pattern
                   collapsed at tk≥1024 from VGPR pressure: 58 µs in step 7);
               (4) chiplet_swizzle wgm=8 chunk=16 — final 2.6% lift from steering
                   consecutive WGs onto the same XCD so A's L1 footprint is reused.

  Patches landed upstream in gemm_universal.py:
    - wire async_buffer_load_lds_addr directly (the AsyncTileLoader path
      assumes attention's wide-K geometry; breaks for tile_m=16, tile_k≥512)
    - per-wave LDS base offset = warp_id * wave_size * BYTES_PER_LANE
      before the DTLA passes — without it, multi-warp DTLA writes collide
      (step 19/20)

  Why no further win is on the table:
    Both kernels live at the sustained-streaming-bf16 ceiling of the
    part (~44% of HBM peak). DSL is *more* bytes-per-µs efficient than
    rocBLAS (44% vs 42%); rocBLAS just transfers fewer total bytes.
    Beating this requires reducing the 33.5 MB byte total — kernel
    fusion or weight quantization — not GEMM-internal tuning.
```

---
Walkthrough: [steps-15-22-multiwarp-chiplet](../10-walkthrough/steps-15-22-multiwarp-chiplet.md). Method: [runbook-discipline](../00-method/runbook-discipline.md).
