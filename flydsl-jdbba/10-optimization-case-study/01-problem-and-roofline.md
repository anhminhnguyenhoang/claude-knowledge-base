---
name: problem-and-roofline
description: "jdbba grouped-GEMM target shapes (HSTU B,D,K,N naming → GEMM dims), the roofline analysis proving it is memory-bandwidth-bound, and why that makes the binding lever cross-block L2 reuse (XCD remap) not compute scheduling"
source: distilled from jdbba optimization on MI355X / gfx950 (2026-06-08..09)
---

# The problem: HSTU jagged_dense_bmm, and why it is memory-bound

Porting Meta's HSTU `jagged_dense_bmm_broadcast_add` (jdbba) to AMD FlyDSL on
MI355X / CDNA4 (gfx950). Per group `b`:
`Out[s:e] = Jagged[s:e] @ Dense[b] + Bias[b]`. bf16 in/out, fp32 accumulate.

## Shape naming trap (HSTU `(B,D,K,N)` ≠ GEMM)

The HSTU benchmark names shapes `(B,D,K,N)` which does **not** match the upstream
Triton docstring's `(B,K,N)`. Map carefully:

| Bench name | Meaning | Standard GEMM dim |
|---|---|---|
| `B` | number of groups / independent GEMMs | batch/group count |
| `N` | `max_seq_len` envelope (M_i ≤ N) | grid M-axis size |
| `D` | width of Jagged = inner dim | **reduction K** |
| `K` | output channel dim | **output N** |

So a group computes `(M_i × D)·(D × K) → (M_i × K)`: **GEMM_M = M_i,
reduction K = D, output N = K_bench**. `D == K_bench` in every headline shape.

### 4 headline shapes (uniform M_i = 7680, a tile-multiple near deployment mean L/B ≈ 7800)

| Shape | B (groups) | D = reduction K | K_bench = output N | regime |
|---|---|---|---|---|
| B1024_D256_K256 | 1024 | 256 | 256 | train, small hidden |
| B1024_D512_K512 | 1024 | 512 | 512 | train, large hidden |
| B120_D256_K256 | 120 | 256 | 256 | inference, small |
| B120_D512_K512 | 120 | 512 | 512 | inference, large |

## Roofline: the kernel is memory-bandwidth-bound

MI355X ≈ 2.5 PFLOP/s bf16, ≈ 8 TB/s HBM → **ridge ≈ 312 FLOP/byte**. jdbba's
arithmetic intensity is **126–248 FLOP/byte < ridge** → memory-bound. Confirmed
on hardware: 32–48% of peak HBM, occupancy 2.8 waves/SIMD (of a 5-wave ceiling),
`MemUnitStalled` 23%, `VALUBusy` 19%, ~25% of peak compute. Compute is idle; the
kernel waits on bytes.

## The design consequences that drive every lever

- **Abundant tile parallelism** (B=120 → ~14.6k tiles; B=1024 → ~125k). There is
  **no occupancy problem to solve** → split-K is unnecessary, `k_batch=1`.
- **Short reduction K** (256/512 → only 4–8 K-iters at BLOCK_K=64). Split-K would
  add atomic overhead for no gain. The short K-loop makes prologue/epilogue + the
  `seq_offsets` read a *larger* share → favor configs that amortize them (this is
  what makes the epilogue store the #1 lever, below).
- **The binding traffic is `Dense[b]` re-reads.** Each group's weight
  (256²–512² bf16 = 128KB–512KB) is reused across **all ~61 M-tiles of its
  group**. That reuse is **cross-block** (same `Dense[b]` across different M-tile
  blocks), which is an **L2 / chiplet** concern — **not** intra-block. So:
  - The winning lever is the **XCD remap** (cross-block L2 reuse).
  - **B-in-LDS regresses** — it targets intra-block reuse, which is low; B is
    already maximally coalesced (`buffer_load_dwordx4`).
  - **Compute scheduling (MFMA atom choice, ping-pong) is the WRONG lever** — the
    machine is not compute- or latency-bound.

> **At bf16 the kernel is at its byte floor; parity-or-slightly-ahead is the
> ceiling.** Five independent experiments (ping-pong, C-strip LDS, W/C autotune,
> skew, async+BLOCK_M+atom) all converge on this. The only lever that can move it
> decisively is **cutting HBM traffic itself** — the XCD remap does this for
> `Dense[b]` reuse without changing numerics; **fp8 weights** (halving the binding
> traffic) is the next real win but changes the numerics contract.

> **Tiny-shape findings did NOT transfer.** An earlier sweep on N=K=128, L≤32k toy
> shapes was launch-bound (CUDA-event wall-clock ~90% host overhead), so its
> per-tile micro-results were meaningless at the real shapes. Only the
> `readfirstlane` epilogue scalarization carried forward. See
> [methodology](../20-methodology/measurement-methodology.md).

---
The idioms that build this kernel: [varlen-jagged-idioms](../00-layout-api-idioms/varlen-jagged-idioms.md). What worked: [winning-levers](02-winning-levers.md). What didn't: [dead-ends](03-dead-ends.md).
