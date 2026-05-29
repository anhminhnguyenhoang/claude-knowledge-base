---
name: skinny-m-decode-gemm
description: Runbook S17.7: the o_proj M=2 skinny decode GEMM tuned to 1.02x rocBLAS on MI355X (condensed). DTLA frees VGPRs to unlock tile_k=1024, multi-warp per-wave LDS offset fix, chiplet swizzle 2.6% win, HBM-efficiency ceiling argument.
source: ck-dsl-optimization-runbook.md (lines 2557-2636)
---

### 17.7 Skinny-M Decode GEMM (o_proj M=2)

Case study from `examples/dsl_o_proj_decode/`. The matmul is Qwen3-8B's
`o_proj` decode: `bf16`, `M=2`, `N=4096`, `K=4096`, MI355X / gfx950, 8 TB/s
HBM peak. rocBLAS sits at 10.10 µs (best) / 42 % HBM efficiency. The
exercise: do whatever it takes to match or exceed rocBLAS, end-to-end
through the runbook loop.

**Final result.** 10.29 µs / 44 % HBM / 1.02 × rocBLAS. Tied at the
HBM-saturation ceiling. Bytes-per-µs is actually *better* than rocBLAS —
rocBLAS just moves slightly fewer bytes. To go further, the lever leaves
the GEMM (fusion with the next op so C never round-trips through HBM).

**Step ladder (selected).** Each row is one runbook lever; numbers are best
of N=10 attempts × 500 iters with the §2.3 hygiene rules.

| Step | Lever | µs    | × rocBLAS | Why |
|----|---|------:|----------:|---|
| 7  | tile_k=1024 without DTLA | 58.0 | 5.74 × | Register pressure: A+B in VGPRs at 1024 K-elements blows the budget, spills, collapses occupancy to 1 wave/CU |
| 13 | + `direct_to_lds=True` | 10.51 | 1.04 × | Frees 32 VGPRs (no A/B round-trip through registers), restoring tile_k=1024 to 2 waves/WG |
| 14 | DTLA cache-hint sweep on A, B | 10.51 | 1.04 × | CACHE_ALL / CACHE_ALL wins on both operands — `CACHE_STREAM` regresses even on the 32 MB B (the 256 CTAs each see a unique N-stripe but consecutive K-tiles within a CTA do reuse L2) |
| 18 | + full DTLA ping-pong prefetch (double-buffered LDS, runtime parity via scf.for iter-arg, `s_waitcnt vmcnt(loads_per_tile)`) | 10.53 | 1.04 × | **Negative result.** Implementation is bit-exact and the prefetch fires, but perf is unchanged. We're not vmcnt-stalled per tile — we're at the system-wide outstanding-loads ceiling already at 1 wave/WG × 256 CUs |
| 19–20 | Fix multi-warp DTLA correctness (per-wave LDS base offset) + sweep wider tiles | 14.0–39 | 1.4–3.8 × | **Negative result.** Multi-warp now correct but slower: wider `tile_n` cuts grid_x proportionally; at `tile_n=32` we drop from 256 → 128 CTAs over 256 CUs and exactly half the GPU goes idle |
| 21 | + `chiplet_swizzle=True, chiplet_wgm=8, chiplet_chunk_size=16` | **10.29** | **1.02 ×** | The 2.6 % win. Consecutive WGs land on the same XCD so each XCD's L2 sees correlated traffic. The other 17 knob combinations cluster between 10.29 and 10.76 µs |

**Lessons reinforced.**

* **The DTLA validator's `cols_per_chunk = halves_per_chunk` assumption is
  attention-specific.** It breaks the moment you try a GEMM tile that isn't
  a single hardware chunk wide (here: `tile_m=16, tile_k=512+`). The fix
  is to call `async_buffer_load_lds_addr` directly with a manually
  distributed pass plan rather than going through `AsyncTileLoader`. See
  `gemm_universal.py:684-820` for the GEMM-shaped form.
* **`async_buffer_load_lds_addr` is wave-level, not lane-level.** Every
  wave writes `wave_size × BYTES_PER_LANE` lane-contiguous bytes starting
  at the wave-uniform `lds_dst`. With multiple waves per WG, each wave
  must target a different LDS slice or they stomp. Add
  `warp_id * wave_size * BYTES_PER_LANE` to the LDS base before the load
  passes (`gemm_universal.py:776-795`). Diagnostic signature: multi-warp
  output is wrong by megabits-of-absdiff, but single-warp is bit-exact and
  the non-DTLA multi-warp path is bit-exact too.
* **Cheap tricks that look like wins are often noise.** Steps 15 and 17
  tested `waves_per_eu` hints over 5 then 20 attempts at 200 then 500
  iters; the apparent 0.07 µs lift at `wpe=4` was inside the stdev. Be
  paranoid about lifts smaller than 1 %; re-run with paired attempts and
  more iters before declaring victory.
* **Prefetch only helps when something is waiting.** §8.2 software
  pipelining and §8.3 `s_waitcnt vmcnt(N)` are real levers when the kernel
  is per-tile vmcnt-stalled. They are *not* levers when the kernel is at
  the system-wide HBM-controller ceiling. Diagnostic: if best ≈ median
  with std < 1 % across the simple-loop variant, more prefetch will not
  help.
* **HBM efficiency is the cap on skinny-M.** With `M=2, N=4096, K=4096,
  bf16`, B alone is 32 MB and each CTA sees a unique N-stripe — there is
  no cross-CTA B-reuse. Total HBM ≈ 33.5 MB; sustained HBM on MI355X for
  streaming bf16 reads tops out near 3.5 TB/s ≈ 44 % of the 8 TB/s
  marketed peak. Both our kernel (44 %) and rocBLAS (42 %) live at this
  ceiling. Beating rocBLAS at this HBM efficiency means reducing total
  HBM bytes — kernel fusion territory (§4.1, §13.5), not GEMM tuning.
* **Trust your rocBLAS reference only in paired hygiene.** Step 22's
  initial back-to-back run measured rocBLAS at 9.63 µs — but only because
  it was running with a warm L2 from our prior DSL kernel. Standalone
  rocBLAS lands at 10.10 µs. Always re-bench the reference in the same
  process and hygiene as the kernel under test.
* **Multi-warp DTLA isn't necessarily a perf lever — but it's a
  correctness fix.** The per-wave-LDS-offset patch in step 20 doesn't help
  *this* shape (CTAs already saturate CUs), but it removes a latent bug
  for any future user who pairs DTLA with `warp_n > 1` (a typical mid-M
  GEMM configuration).

**Reproducible commands.**

```bash
cd examples/dsl_o_proj_decode
python scripts/13_dtl.py       # DTLA + cache-hint sweep
python scripts/18_dtl_prefetch.py
python scripts/19_multiwarp_probe.py   # isolates the wave-stomping bug
python scripts/21_chiplet.py    # the 2.6 % win
python scripts/22_confirm_winner.py    # 20 × 1000 paired vs rocBLAS
```

---
This is the condensed form of the full 22-step worked walkthrough in the sibling topic [../../ck-dsl-runbook/](../../ck-dsl-runbook/README.md). DTLA/chiplet knobs: [knob-catalog-and-sweep](../30-autotuning/knob-catalog-and-sweep.md) (S12.1.F/L).
