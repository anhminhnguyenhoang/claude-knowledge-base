---
name: chiplet-xcd-remap
description: Reorder GPU work blocks across AMD CDNA chiplets (XCDs) for cache reuse. Use when optimizing a GEMM/conv/attention kernel on MI300/MI325/MI350/MI355X that has cross-block data reuse and shows a low L2 hit rate, or when asked to apply chiplet/XCD grid swizzle, block-ID remap, or "Algorithm 1" launch-time scheduling.
argument-hint: [kernel file or problem shape]
---

# Chiplet (XCD) Work Reordering

AMD CDNA3/4 GPUs (MI300X/MI325X/MI350X/MI355X) are **8 chiplets (XCDs)** glued
together. Each XCD has its own **private, fast L2** (4 MB on MI355X, ~3× the
bandwidth of the shared cache); a single **shared LLC** sits between all 8 L2s
and HBM. The hardware scatters thread blocks round-robin across the 8 XCDs by
default, which wrecks L2 reuse.

The fix is a **launch-time block-ID remap**: ~20 lines of integer math at the
top of the kernel that reassigns which output tile each block computes. **No
change to the compute body, no extra memory.** Measured **+19% bandwidth** on a
9216³ BF16 GEMM (HipKittens Algorithm 1).

This is the single best cost/benefit lever for large, reuse-heavy kernels on
these chips.

## When to apply (and when not to)

```
Does the kernel read input that could be reused across blocks?
   ├─ No  (elementwise, pure pointwise) → SKIP. Nothing to cache.
   └─ Yes (GEMM, conv, attention)
        ├─ Problem small (a few hundred CUs of work) → marginal; round-robin damage is bounded.
        └─ Problem large (10k+ blocks) → APPLY. This is where it pays.
```

First confirm the symptom: a **low L2 hit rate (~55%) with high LLC hit rate**
on a reuse-heavy kernel is the signature that round-robin scheduling is hurting
you. Measure before and after.

## The mechanism: default scheduling is bad

Hardware routes block `xy` to XCD `xy % 8`. So XCD 0 runs blocks
`{0, 8, 16, 24, ...}` — every 8th block in launch order. Combined with
row-major launch, each XCD ends up working on a **scattered set** of output
tiles that share neither a row of A nor a column of B consistently → L2 misses.

You defeat this by **using the round-robin against itself**: emit block IDs in
stride-8 so consecutive *logical* blocks land on the *same* XCD.

## The two knobs

The remap has exactly two tunable parameters. Set both — each alone underperforms.

### Knob C — chunk size (group blocks onto one XCD)
Force `C` consecutive logical blocks onto the same XCD so they share data in
that XCD's fast L2.
- **Too small** (C=1) → back to the default round-robin, no grouping.
- **Too large** → each XCD works on totally separate input slices → the shared
  LLC sees no overlap and **starves**.
- **C trades LLC reuse for L2 reuse.**

### Knob W — window height (walk in 2D, not row-major)
Within an XCD's chunk, visit output tiles in **vertical windows W rows tall**
instead of row-by-row. This makes each consecutive group of blocks reuse
**both** a row tile of A **and** a column tile of B — exactly what L2 wants.
- **W=1** → degenerates to row-major (no gain).
- **W = num_rows** → maximizes B-column reuse, loses A-row reuse.
- **Moderate W (4–8)** → balances both.

The two compose **hierarchically**: Phase 2's windowed walk is applied to the
remapped ID from Phase 1, so windowing happens *within* each XCD's chunk.

## Reference implementation (HipKittens Algorithm 1)

Done at the very top of the kernel, before any compute:

```cpp
// Phase 1: XCD grouping — invert the hardware round-robin so C consecutive
// logical blocks land on the same XCD.
//   nXCD = 8 on MI300/MI325/MI350/MI355X
uint32_t xcd       = xy % nXCD;
uint32_t local     = xy / nXCD;
uint32_t chunk_idx = local / C;
uint32_t pos       = local % C;
uint32_t xy_g      = chunk_idx * (nXCD * C) + xcd * C + pos;

// Phase 2: hierarchical windowed traversal — walk xy_g in W-row vertical windows.
uint32_t num_rows      = M / BLOCK_M;
uint32_t num_cols      = N / BLOCK_N;
uint32_t tids_per_grp  = W * num_cols;
uint32_t group_id      = xy_g / tids_per_grp;
uint32_t first_row     = group_id * W;
uint32_t win_h         = min(num_rows - first_row, W);  // last window may be short
uint32_t l             = xy_g % tids_per_grp;
uint32_t row           = first_row + (l % win_h);
uint32_t col           = l / win_h;
// use (row, col) as the real output-tile coordinates; A[row,:] @ B[:,col]
```

```cpp
__global__ void gemm_kernel(...) {
    uint32_t raw_xy = blockIdx.y * gridDim.x + blockIdx.x;
    auto [row, col] = xcd_windowed_remap(raw_xy, W, C, M, N, BLOCK_M, BLOCK_N);
    // ... unchanged compute body using (row, col)
}
```

## Recommended starting values & proven sweet spots (MI355X)

| Situation | (W, C) | Why |
|---|---|---|
| **Start here, then sweep** | (4, 16) or (8, 32) | Safe balanced region |
| Small problem (≈9216³) | **W=5, C=25** | Measured 75% L2 / 93% LLC / 18.3 TB/s (+19%) |
| Large problem (≈14592³) | **W=8, C=64** | More reuse to amortize → bigger chunk |

Rule of thumb: bigger problems can afford bigger `C`. Aim roughly for
`C ≈ W × (columns per window)` so an XCD's chunk fits in a couple of windows.
The paper's general sweet-spot framing: "8×4 or 4×8 L2 tiles" → L2 ~75%,
LLC ~90%.

## The L2-greedy trap (the one thing not to do)

**Maximizing L2 hit rate alone makes you slower than doing nothing.**

| Config (9216³) | W | C | L2 | LLC | Bandwidth |
|---|---|---|---|---|---|
| Row-major baseline | — | — | 55% | 95% | 15.1 TB/s |
| **L2-greedy** | 7 | 216 | **79%** | **24%** | 14.9 TB/s ← *worse than baseline* |
| Balanced | 5 | 25 | 75% | 93% | **18.3 TB/s** |

The L2-greedy config has the **highest L2 hit rate** and the **worst
performance** — starving the LLC dominates. **Tune for the bandwidth number,
never the cache-hit number. Watch L2 AND LLC together; never sacrifice one for
the other.**

## CK DSL equivalent (if using ck_dsl / Python tooling)

The same optimization ships as ready-made knobs — you don't hand-write the remap:

```python
chiplet_swizzle   = True   # off by default — turn it ON
chiplet_chunk_size = 64    # the C knob (XCD round-robin chunk size)
chiplet_wgm        = 8     # super-tile grouping (the windowing knob)
chiplet_num_xcds   = 8     # MI300X/MI325X/MI350X all have 8
```

- Helper for hand-written kernels: `helpers/grid.py::chiplet_transform_chunked`
- Constants: `helpers/grid.py::NUM_XCDS_MI300X / MI325X / MI350X` (all 8 today)

## Verification workflow

1. **Before**: measure L2 hit, LLC hit, and bandwidth (rocprof). Confirm the
   low-L2 / high-LLC signature.
2. Apply the remap with a starting `(W, C)`.
3. **After**: re-measure all three. The bandwidth number is the verdict.
4. Sweep `(W, C)` around the starting point. **Reject any config that raises L2
   by starving LLC unless total bandwidth went up.**
5. The right `(W, C)` is problem-size-dependent — re-tune for materially
   different shapes.

## One-sentence takeaway

> Defeat the hardware's round-robin with a launch-time block-ID remap: knob **C**
> groups consecutive blocks onto one XCD for L2 reuse, knob **W** walks that
> chunk in 2D windows so each group reuses both a row of A and a column of B —
> and always judge by bandwidth, because maximizing L2 alone starves the shared
> LLC and loses.

Sources: HipKittens (arXiv 2511.08083) Algorithm 1; CK DSL Optimization Runbook §12.1.L / §21.5.
