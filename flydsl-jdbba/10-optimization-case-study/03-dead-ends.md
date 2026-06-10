---
name: dead-ends
description: "jdbba optimization dead-ends, each verified on a clone with cos=1.0: bigger tiles, STAGES_A=3, B-in-LDS, packed grid, ping-pong/4-wave interleave, finer W/C autotune, C-shuffle LDS shrink, M register-tiling, async-copy+BLOCK_M256, 32x32 atoms, DirectToLDS — and WHY each failed at a memory-bound kernel"
source: distilled from jdbba follow-up experiments 1-8 (MI355X / gfx950, 2026-06-09..10)
---

# Dead-ends (verified — do not repeat)

Each was changed in isolation on a clone (production kernel untouched), verified
cos=1.0, measured with rocprofv3/do_bench. **All are negative for folding.** The
unifying reason: the kernel is **memory-bandwidth-bound at the bf16 byte floor**
(see [problem-and-roofline](01-problem-and-roofline.md)), so any lever that does
not cut HBM traffic buys nothing — and several *raise* traffic or *cost*
occupancy.

## Tiling / pipeline structure

| Lever | Why it failed |
|---|---|
| `BLOCK_M = 64 / 256`, `BLOCK_N = 256` | VGPR blowup → occupancy collapses to 1 wave/SIMD. Default **128/128 is the sweet spot even when compute would want bigger**. |
| `STAGES_A = 3` | No-op — the ping-pong only uses 2 buffers; the 3rd is allocated, never read. Real 3-stage needs a `run_pipeline_stage` restructure. |
| Dense (B) staged through LDS | Regresses — B is already maximally coalesced (`buffer_load_dwordx4`) and **intra-block** B-reuse is low. The real B reuse is **cross-block** (L2/chiplet → that's the XCD remap, a different lever). |
| Packed / persistent grid on **uniform** data | Irrelevant — uniform `M_i = max_seq_len` has zero tail-tile waste. (Only helps the skewed deployment; see the persistent kernel under winning-levers.) |

## Latency / occupancy (the kernel is neither latency- nor occupancy-bound)

- **Ping-pong / 4-wave interleave: ruled out by the occupancy probe.** The kernel
  is occupancy-*resource*-limited (32KB LDS + 84 VGPR cap ~5 waves/SIMD, only 2.8
  achieved), **not latency-bound**; more waves can't fit and can't fix a 49% L2
  miss.
- **C-shuffle LDS shrink (N-strips): no-op.** Premise false — the A double-buffer
  *staging* LDS (`BLOCK_M·BLOCK_K·STAGES_A·2`) is ≥ the epilogue C tile at default
  tiling, so it sets the `max()` and shrinking the epilogue leaves launched
  LDS/block unchanged → occupancy can't rise. To lift occupancy you must shrink
  the **binding** resource (A-staging LDS via STAGES_A/BLOCK_K, or VGPR), not the
  epilogue. (S=4 isn't expressible — MFMA-16x16 C-frag N-repeat granularity = 64.)
- **async-copy A + BLOCK_M=256 + larger atom (the bundled roofline-legal probe).**
  Async copy works (cos=1.0) and *did* drop VGPR 84→76 — but occupancy barely
  moved (2.80→2.85) and time got ~3% worse. **Decisive proof the kernel is NOT
  VGPR/occupancy-limited: freeing VGPR bought zero waves and zero time.**
  BLOCK_M=256 became *viable* with async but doesn't help (higher AI yields
  nothing when traffic-bound; a 64KB tile halves concurrent blocks). The larger
  MFMA atom was correctly gated out — a bigger atom only burns more accumulator
  VGPR (32x32x16 = 16 f32/lane vs 4) for zero gain when memory-bound.

## Autotune / micro-tuning

- **Finer (W,C) sweep (uniform):** does not beat the defaults beyond the ~0.5%
  noise floor. The D512 C-curve is a flat plateau for C≥32; W=8 is flat-optimal.
  Only reproducible signal: **W=16 on the D256 shapes (~1.2–1.7%)** — real but
  marginal, not folded.
- **M register-tiling** (raise B reuse without growing LDS): does **not** cut HBM
  traffic (DRAM reads actually rise); the small D512 win is a block-count/MFMA-issue
  effect and D256 regresses.

## Copy-path micro-opts

- **DirectToLDS-for-A (standalone):** regression 0.80–0.97× all shapes. The
  immediate `s_waitcnt vmcnt(0)` surrenders DTLA's latency-hiding, and
  `BufferCopyLDS128b` **silently breaks the `Swizzle(3,3,3)` LDS layout** at
  BLOCK_K=64 (cos≈0.13 until switched to a plain ordered layout → s2r bank
  conflicts). ISA confirmed the mechanism worked (ds_write_b128 24→0, VGPR 194→168)
  but the trade-offs eat it. Only revisit paired with prefetch (deferred vmcnt) +
  a swizzle-free `ds_read_tr` transpose-read.

## The one reversal worth recording

An early experiment on a **pre-fix clone** found the XCD remap *regressed* ~2%
under skew. **Re-measured on the current kernel** (after the bounded-A fix +
16x16x32 atom) the remap is ~2% *faster* under skew on D512 and neutral on D256.
The old "regression" was the **unbounded-A OOB perturbing timing**, not the remap.
Lesson: **re-measure a surprising negative on the *current* kernel before trusting
it** — a prior bug can masquerade as a lever's effect.

---
What worked instead: [winning-levers](02-winning-levers.md). The measurement discipline that caught the reversal: [methodology](../20-methodology/measurement-methodology.md).
