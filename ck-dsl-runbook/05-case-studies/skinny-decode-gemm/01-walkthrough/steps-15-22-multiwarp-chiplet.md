---
name: steps-15-22-multiwarp-chiplet
description: Steps 15-22: the multi-warp DTLA correctness bug (wave-uniform lds_dst collision) and its per-wave LDS-offset fix, perf-neutral ping-pong prefetch, and the final 2.6% chiplet_swizzle lift confirmed at N=60.
source: ck-dsl-gemm-skinny-decode-README.md (lines 518-626)
---

## Step 15–17 — wave geometry & `waves_per_eu` retest (`15_…`, `16_…`, `17_…`)

Once tk=1024 + DTLA + CACHE_ALL/CACHE_ALL was landed, the obvious next
ask was "more lanes per WG should expose more in-flight global loads".
Step 15 swept `warp_m × warp_n ∈ {1×2, 1×4, 2×2}` at the winning tile.
Step 16 widened `tile_n` (32, 64) to feed those extra waves real work.
Step 17 re-confirmed `waves_per_eu` now that the kernel is no longer
single-wave.

Every multi-warp variant in step 15 produced **garbage output**
(`max_abs ≈ 5 × 10⁴`) and step 16 was nonsense as a result. That made
multi-warp DTLA the suspect, not "DTLA is bad".

## Step 18 — DTLA ping-pong prefetch (`18_dtl_prefetch.py`)

> Runbook §12.1.D (pipeline depth), §17.4 (compose-or-not).

Plumbed `TraitSpec.dtl_prefetch=True`: double-buffered LDS, scf.for
iter-arg carries the live half-index, the prologue issues the first
DTLA into half-0 while the steady-state loop drains half-(i⊕1) and
fires DTLA into half-i. The `s_waitcnt vmcnt(loads_per_tile)` between
the two DTLA passes had to be tuned by hand (gfx950 caps vmcnt at 6
bits = 63).

Result: bit-exact (max_abs=0). Perf identical to non-prefetch
(10.51 µs both). The 1-wave/CTA kernel is HBM-bound — adding a second
in-flight tile doesn't help because the HBM controllers are already
the bottleneck. Kept as `dtl_prefetch` for future shapes where the
bound shifts.

## Step 19 — isolate the multi-warp DTLA bug (`19_multiwarp_probe.py`)

Four-cell matrix: `{single-warp, multi-warp} × {DTLA off, DTLA on}`,
all bit-exact ref check. Result before the patch:

| | DTLA off | DTLA on |
|---|---:|---:|
| 1 warp  | max_abs = 0 | max_abs = 0 |
| 2 warps | max_abs = 0 | **max_abs ≈ 5.2 × 10⁴** |

The bug is **specifically** in multi-warp DTLA. Diagnosis:
`async_buffer_load_lds_addr` is a wave-level intrinsic and writes
into a **wave-uniform** `lds_dst`. Every wave was given the same
`a_lds_par_base` / `b_lds_par_base`, so they stomped each other.

Patch in `gemm_universal.py:emit_load_phase`: compute a per-wave
LDS base offset (`warp_id × wave_size × BYTES_PER_LANE`) and pass
`a_lds_wave_base` / `b_lds_wave_base` to both A- and B-load passes
(must fix both — first fix only patched the A-loop and step 19 still
failed on the B side).

Re-running step 19 post-patch: all four cells max_abs = 0.

## Step 20 — multi-warp DTLA, now correct (`20_dtl_multiwarp.py`)

Resweep `(tm, tn, tk, warp_m, warp_n)` with multi-warp finally legal:
`16×32×512 w1×2`, `16×64×256 w1×4`, `32×64×256 w2×4`, etc.

Every wider tile was slower. Best multi-warp finish (`tm16 tn32 tk512
w1×2`): ~14.0 µs vs single-warp 10.5 µs. Mechanism: wider tile_n cuts
the grid in half (`256 → 128 CTAs` over `256` CUs), so half the CUs
idle. At `1 wave/WG × 256 CUs` the kernel already saturates the HBM
controllers. **Multi-warp was a correctness fix, not a perf lever** on
this shape.

## Step 21 — chiplet_swizzle (`21_chiplet.py`)

> Runbook §12.1.L (chiplet swizzle).

MI355X has 8 XCDs; the default WG dispatch order scatters consecutive
CTAs across XCDs so each XCD's L2 sees uncorrelated traffic.
`chiplet_swizzle` remaps WGIDs via `chiplet_aware_super_tile_dynamic`
so consecutive WGs land on the same XCD. At `M=2` the same 16 KiB A
tile is reused by every CTA in a M-row — exactly the cross-CTA reuse
this knob targets.

Sweep `wgm ∈ {2,4,8,16}` × `chunk ∈ {16,32,64,128}` at the locked
DTLA winner, 10 attempts × 500 iters:

| variant | best µs | vs rocBLAS |
|---|---:|---:|
| baseline (no chiplet) | 10.533 | 1.04× |
| **chiplet wgm=8 chunk=16** | **10.292** | **1.02×** |
| chiplet wgm=4 chunk=16 | 10.295 | 1.02× |
| chiplet wgm=16 chunk=16 | 10.299 | 1.02× |

**2.6 % real lift.** Top three wgm values cluster within 7 ns at
chunk=16; wider chunks regress (the chunk has to be small enough that
A's reuse window fits inside an XCD's residency).

## Step 22 — high-confidence confirmation (`22_confirm_winner.py`)

20 attempts × 1000 iters × 3 interleaved rounds (N=60 each for chiplet,
baseline, paired rocBLAS), then a standalone rocBLAS probe so the
reference isn't cache-warm-biased.

| label | best µs | median µs | std |
|---|---:|---:|---:|
| chiplet winner | 10.292 | 10.45 | 0.08 |
| baseline (no chiplet) | 10.560 | 10.73 | 0.09 |
| rocBLAS (standalone, fair) | 10.10 | 11.01 | — |
| rocBLAS (paired, hot L2) | 9.63 | 9.71 | 0.04 |

The paired rocBLAS reads ~5 % faster than its standalone reading
because the chiplet/baseline kernels in the same loop pre-warm the
weight tile's L2 footprint. The honest reference is the standalone
probe: **DSL 10.29 µs vs rocBLAS 10.10 µs = 1.02× — within the noise
floor.** DSL is actually **more bytes-per-µs efficient** than rocBLAS
(44 % vs 42 % HBM); rocBLAS just streams slightly fewer total bytes.

---
Prev: [steps-12-14-direct-to-lds](steps-12-14-direct-to-lds.md). Chiplet background: [hipkitten chiplet-scheduling](../../../../hipkitten/01-paper/chiplet-scheduling.md). Outcome: [ceiling-and-followups](../02-conclusion/ceiling-and-followups.md).
