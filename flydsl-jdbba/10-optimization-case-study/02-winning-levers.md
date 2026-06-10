---
name: winning-levers
description: "The five jdbba optimization levers that worked: LDS C-shuffle epilogue vectorization, shape-dependent BLOCK_K, XCD chiplet block-ID remap (HipKittens Alg 1), the CDNA4 16x16x32 bf16 MFMA atom, and the XCD-aware persistent visiting order for skew — with mechanism, magnitude, and gotchas"
source: distilled from jdbba_dense_bmm_gen.py + persist_dev optimization (MI355X / gfx950, 2026-06-08..10)
---

# Winning levers (folded into production)

Device time is the metric throughout (rocprofv3 `--kernel-trace`, p10; or do_bench
where noted). Each lever was changed in isolation on a clone, verified cos=1.0,
measured, then re-verified in one interleaved sweep before promotion (GPU-clock
drift between separate runs is real).

## 1. LDS C-shuffle epilogue vectorization — +6–17% (the largest win)

The MFMA C accumulator is **M-major per lane**: a lane's contiguous fragment
elements map to different output *rows* (stride N in global), so a vectorized
N-contiguous store straight from the fragment is impossible. The baseline emitted
**64 scalar `buffer_store_short` per thread**; a store-deletion diagnostic showed
the store alone cost **38–62% of runtime** at these memory-bound shapes.

Fix: route C through LDS to transpose the layout — write the bf16+bias fragment to
a row-major shared C tile in its natural MFMA layout (**reusing the A-staging LDS,
no extra smem**), barrier, then re-read N-contiguous (8 bf16/thread) and store
`buffer_store_dwordx4`. **64 narrow stores → 8 wide stores.** This is the general
"epilogue store vectorization is often the single largest win for a kernel that
already has a good main loop" rule. ISA proof: `buffer_store_short` disappears,
`buffer_store_dwordx4` appears.

## 2. Shape-dependent BLOCK_K — ~+4% on K=256

`block_k = 128 if reduction_K <= 256 else 64`. K=256 prefers a 2-iter K-loop
(BLOCK_K=128, fewer barriers); K=512 prefers the deeper BLOCK_K=64 pipeline for
occupancy. **`BLOCK_K=256` is UNSAFE** — the 2-stage double-buffer epilogue
silently mis-accumulates a single K-tile (cosine masked a 123% relative error;
this is why correctness must use cosine *and* mean-signed-error, not just cosine).

## 3. XCD (chiplet) block-ID remap — ~+5% on D=512 (HipKittens Algorithm 1)

MI355X / CDNA4 has **8 XCDs**; the hardware routes raw block id `xy` to XCD
`xy % 8`. Co-locate a group's M-tiles (which share that group's `Dense[b]`) onto
**one** XCD's private L2 by inverting that round-robin, so `Dense[b]` is not
re-fetched from HBM into all 8 L2 slices. Measured on B1024_D512: **L2 hit
49%→76%, DRAM read requests −67%, device time −5.3%**.

Knobs:
- `W = 8` (window height) — a **flat optimum** across shapes.
- `C` (chunk size) is **weight-size-dependent**: `XCD_C_SMALL_K=32` for K≤256,
  `XCD_C_LARGE_K=120` for K>256 (autotuned plateau). **Small K regresses with a
  large C** — the *L2-greedy / LLC-starvation trap*: maximizing one XCD's L2 reuse
  while starving the shared LLC loses.
- The remap identity-maps the non-`(8·C)`-divisible tail so it stays an exact
  bijection for any C/W → cos=1.0 always.

## 4. CDNA4 16x16x32 bf16 MFMA atom — +4–7%, bit-exact

Swap the `MFMA(16,16,16)` atom for `MFMA(16,16,32)` on gfx95* (gfx942 falls back
to 16x16x16). Same 4 fp32 accumulators/lane, **half the MFMA issue** (17.6B → 8.8B
instructions) — an *issue-efficiency* win, not occupancy. Bit-exact. Wiring:
K-permute `(4,4,2),(1,8,4) → (8,4),(1,8)`; the inner-loop fragment K index
flattens `(None, iter) → iter`. Mirrors `preshuffle_gemm_v2`'s `use_mfma_k32`.
Gated behind `_use_mfma_k32` (auto-on for gfx95*).

> **Stacking (3)+(4):** B1024_D512 went 6728 → 6377 (remap) → 5951 µs
> (remap+mfma32) ≈ **1.13× vs pre-remap**. These are what moved the kernel from
> parity to ahead on the D=512 shapes.

## 5. XCD-aware persistent visiting order — +2–5% on skew (the varlen lever)

For **skewed/varlen** deployment (`M_i = max_seq_len·U⁴`, ~27% empty groups), a
static max-seq-len grid early-exits most blocks. The persistent problem-visitor
kernel (below) fixes that waste, but the *first* persistent version forfeited the
XCD L2 reuse — its blocks claim linear tile ids `blk_id, blk_id+grid, …`; the grid
(512) is a multiple of 8 so a block's XCD `== wi % 8` for every id it claims, but
linear ids are **group-major** (a group's tiles are a contiguous run sharing
`Dense[b]`) → consecutive ids go to consecutive blocks → consecutive XCDs → each
`Dense[b]` is pulled into all 8 L2 slices again.

Fix: **decouple the visiting index `v` (strided loop var, `v % 8 ==` physical XCD)
from the tile id `wi`.** Apply HipKittens phase-1 chunking to `v → wi` so a run of
`C` consecutive group-major tile ids co-locates on one XCD. Because `total_tiles`
is device-read, the prefix is computed **at runtime** (vs compile-time in the
static kernel) and `readfirstlane`'d to stay uniform.

Measured (do_bench, kernel-only, 3 seeds): **+2–5% over baseline persist_dev on
every skew shape.** It **flips both D256 cells from behind to ahead of Triton**
(B120_D256 +3–6%, B1024_D256 +1–4%). D512 stays ~13–20% behind Triton — the XCD
order helps ~3–7% there but cannot close it: **D512 is at the bf16 byte floor, so
only fp8 weights close the D512 skew gap.** Sweet spot `C ∈ {16, 32}` (skew chunks
are short — mean ~9 M-tiles/group). A scheduling win, not a bandwidth win.

## The persistent problem-visitor kernel (the skew path)

Replaces the static `(ceil(max_seq_len/BLOCK_M)·N_BLOCKS, 1, n_groups)` grid with
a fixed persistent grid that pulls **only occupied tiles**. The key is doing it
with **zero host sync**:

- Build a per-group cumulative occupied-tile prefix `CUM[n_groups+1]` **on device**
  with a pure-torch expression (`mb = so[1:]-so[:-1]; tiles =
  ceil(mb/BLOCK_M)*N_BLOCKS; CUM = [0, *tiles.cumsum()]`) — no `.cpu()`/`.item()`.
  (Design A's host `seq_offsets.cpu()` work-list build erased the win end-to-end;
  this is Design B-1.)
- Each block reads `total_tiles = CUM[n_groups]` into a register
  (`buffer_load` + `readfirstlane`) and strides the persistent loop.
- Map linear tile id → `(off_b, block_m_idx, block_n_idx)` by a **compile-time
  unrolled binary search** over `CUM` (log₂(n_groups) loads/tile; a linear scan
  over 1024 groups was the kernel-only bottleneck). Empty groups have zero-width
  `CUM` intervals so they are skipped automatically; group-major linear order
  preserves L2 locality for free.

Routed to by `jagged_dense_bmm_dispatch_v2.py` only for non-uniform + large/D512
shapes (it regresses ~20–44% on uniform — it forfeits the static grid's XCD remap).

---
Why these and not others: [problem-and-roofline](01-problem-and-roofline.md). What was ruled out: [dead-ends](03-dead-ends.md). How each was measured: [methodology](../20-methodology/measurement-methodology.md).
