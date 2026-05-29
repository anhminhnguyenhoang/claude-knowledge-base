---
name: work-decomposition
description: Runbook S5: grid / block / wave / thread mapping. Chiplet swizzle for CU balance, wave64 block sizes, MFMA lane-to-output mapping, and the canonical K-pack lane-mapping bug (validate at 1e-3).
source: ck-dsl-optimization-runbook.md (lines 728-801)
---

## 5. Work Decomposition

### 5.1 Grid Mapping

- Map large independent dimensions to grid axes.
- Fold small batch / head / group dimensions into the x-axis if it
  improves scheduling. Per-instance `*_grid` helpers own this
  (e.g. `gemm_universal.gemm_universal_grid`).
- Avoid putting too much work only in grid-z if it reduces scheduler
  interleaving.
- Balance blocks across CUs. The MI350X chiplet swizzle in
  `helpers/grid.py::chiplet_transform_chunked` (with
  `NUM_XCDS_MI300X/MI325X/MI350X == 8`) can help.
- Ensure enough CTAs for occupancy.
- Avoid too many tiny CTAs with high launch overhead.
- Avoid too few huge CTAs that underfill the GPU.
- Consider persistent CTAs for irregular work
  (`helpers/persistent.py::persistent_tile_loop`).

### 5.2 Block Mapping

- Choose threads per block to match wave count and occupancy.
- Common wave64 choices: 64, 128, 256, 512, 1024.
- Block size is encoded in the spec (`tile_m`, `tile_n`, `tile_k`,
  `warp_m`, `warp_n`, etc.) and verified at lowering by
  `kernel.attrs["max_workgroup_size"]`.
- More waves per block can improve data sharing but reduce occupancy.
- Fewer waves per block can improve occupancy but reduce reuse.
- Sweep wave count via `probe_config_sweep.py` with
  `num_warps ∈ {1, 2, 4, 8}` overrides.
- Sweep groups / heads per block.
- Sweep spatial columns per block.
- Sweep rows per block (`block_m_per_warp` in the 2D attention spec).
- Keep hot-loop per-block work high enough to amortize barriers.

### 5.3 Wave Mapping

- Decide what each wave owns. The `helpers/geometry.py::WarpGrid`
  type packs tile + warp grid + bound `tid / lane / warp_* / block_*_off`
  SSA into one immutable view.
- Map lanes to contiguous memory where possible.
- Map lanes to MFMA operand layout exactly. The
  `MfmaAtom.lane_to_output(b, lane, i)` helper documents the lane
  mapping per atom shape.
- Avoid lane mapping that forces many shuffles.
- Avoid wasted lanes on small dimensions.
- Consider multiple independent small problems per wave.
- For `4x4x4` MFMA, use lane batching.
- For `16x16x16` MFMA, align `lane % 16` and `lane / 16` to matrix
  dimensions.
- For `16x16x32` MFMA, understand K-lane packing before using it.

For `mfma_f32_16x16x32_f16` on AMD CDNA, lane `(c4 = lane / 16)`
holds K elements `[c4*8 : c4*8 + 8]`, not a flat concatenation of two
4-element halves. A common bug when folding two filter columns
`S=0, S=1` into K=32 is to pack `[S=0 ch 0..3, S=1 ch 0..3]` per lane.
The correct mapping is `[S=0 ch 0..7]` for `c4=0`, `[S=0 ch 8..15]`
for `c4=1`, `[S=1 ch 0..7]` for `c4=2`, `[S=1 ch 8..15]` for `c4=3`.
The wrong packing compiles, runs, and validates within `1e-2` but
fails at `1e-3` (`max_abs ~ 5e-3`, ~10 % of elements bad). Always
validate K-pack lane mapping at `1e-3`, not just at `1e-2`.

### 5.4 Thread Mapping

- Assign vectorized global loads to contiguous lanes
  (`helpers/loads.py::lane_contiguous_descriptor`).
- Assign stores to contiguous lanes.
- Separate compute lanes from store lanes only if it improves
  coalescing.
- Use inactive lanes deliberately, not accidentally.
- Avoid dynamic modulo/divide in hot loops when compile-time
  decomposition is possible. The DSL's `IRBuilder.static_for` /
  `unroll` / `static_if` are Python-time-only.
- Precompute per-thread offsets outside loops.

---
Next: [memory-hierarchy](memory-hierarchy.md). MFMA operand layout: [matrix-instructions](matrix-instructions.md). Chiplet knobs: [knob-catalog-and-sweep](../30-autotuning/knob-catalog-and-sweep.md) (S12.1.L); arch caps: [target-architecture-gfx950](../60-reference/target-architecture-gfx950.md).
