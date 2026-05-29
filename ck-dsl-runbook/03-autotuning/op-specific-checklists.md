---
name: op-specific-checklists
description: Runbook S13: per-operation checklists for GEMM, convolution, attention, reduction, and fused ops — the questions to ask before and during tuning each family.
source: ck-dsl-optimization-runbook.md (lines 1747-1820)
---

## 13. Operation-Specific Checklists

### 13.1 GEMM Checklist

- Is layout transposed or not? (`UniversalGemmSpec.layout`)
- Are operands contiguous along the vectorized dimension?
- Does tile shape match MFMA? (`MfmaAtom.shape`)
- Is K loop long enough?
- Are A/B staged through LDS?
- Are loads vectorized? (`probe_intrinsic_counts.py`:
  `global.load`/`buffer.load` count vs MFMA count)
- Is epilogue vectorized? (`probe_isa_inspect.py`:
  `buffer_store_dwordx2`/`_x4` vs `buffer_store_short`)
- Is split-K needed?
- Is Stream-K needed?
- Is persistent scheduling helpful?
- Are scales / bias fused efficiently? (`FusedEpilogue`)

### 13.2 Convolution Checklist

- Direct vs implicit GEMM decision (see Section 4.2).
- `N_gemm`, `K_gemm`, reduction length.
- Channel count small or large.
- Filter size (3×3 vs 5×5 vs 7×7).
- Padding / tails.
- Row streaming.
- Circular accumulators.
- Weight residency (registers vs LDS).
- Input LDS reuse.
- Q / spatial tiling.
- Group tiling.
- Output store mapping.
- Swizzle on `(spatial, C8)`.

### 13.3 Attention Checklist

- Tile Q / K / V.
- Online softmax (`OnlineSoftmaxState`).
- Mask handling (`causal_mask`, `sliding_window_mask`).
- Page table overhead (`transforms.indirect + unmerge`).
- KV cache layout.
- Split sequence / head (2D vs 3D).
- Register pressure from accumulators.
- Shared memory capacity.
- Coalesced V/O stores.
- LSE output.
- 2D vs 3D selection heuristic (`select_2d_config`, `use_2d_kernel`).
- ALiBi / QQ-bias paths exercised (per
  `notes/ATTENTION_PARITY_REPORT.md` 2D path was failing C9/C10/C11
  until the score-block fix landed).

### 13.4 Reduction Checklist

- Axis length.
- Rows per block.
- Vector width.
- Warp-level vs block-level (`block_lds_reduce` vs warp-only).
- Multi-pass.
- Numerical stability.
- Atomic or no atomic.
- Store width.

### 13.5 Fused Op Checklist

- Does fusion reduce memory traffic?
- Does fusion increase VGPR too much? (`probe_occupancy.py`)
- Can the epilogue be fused without scalarizing stores?
- Are scale / bias loads coalesced?
- Are activation approximations acceptable?
- Is intermediate precision visible?
- The DSL fusion pipeline (`helpers/fuse.py`) is the place; it
  routes through `FusionLegalizer.legalize(graph)` to verify
  LDS budget, dtype, vector width, and atomic-region rules before
  lowering.

---
Family instance tables: [algorithmic-mapping](../02-levers/algorithmic-mapping.md). Full knob list: [knob-catalog-and-sweep](knob-catalog-and-sweep.md). Failure modes per op: [failure-modes](../04-failure-reporting/failure-modes.md).
