---
name: algorithmic-mapping
description: Runbook S4: pick the kernel before the knobs. GEMM / conv / attention / reduction / transpose / gather family instance tables, plus the implicit-GEMM VGPR-tax explainer.
source: ck-dsl-optimization-runbook.md (lines 492-724)
---

## 4. Algorithmic Mapping Choices

> **Hardware Resource Consumption Warning**: Different algorithmic
> approaches for the same operation can have dramatically different
> hardware resource requirements (VGPRs, SGPRs, LDS) even when they
> achieve similar compute throughput. Always profile resource use
> alongside performance — `probe_occupancy.py` is the per-variant
> answer.

### 4.1 GEMM Family

| Variant | Instance |
|---|---|
| Plain GEMM | `instances/gemm_universal.py::UniversalGemmSpec` |
| Batched GEMM | `instances/batched_gemm.py::BatchedGemmSpec` |
| Strided batched | same; layouts in the spec |
| Grouped GEMM | `instances/grouped_gemm.py::GroupedGemmSpec` |
| Persistent GEMM | `helpers/persistent.py` + `instances/streamk_gemm.py` |
| Stream-K | `instances/streamk_gemm.py::StreamKGemmSpec` |
| Split-K | `b.global_atomic_add_f32` available; spec field WIP |
| Preshuffled GEMM | `helpers/preshuffle.py::PreshuffleBSpec` |
| Block-scaled GEMM | `instances/block_scale_gemm.py::BlockScaleGemmSpec` |
| Sparse GEMM | not yet implemented |
| Quantized GEMM | block-scaled / mx GEMM + `helpers/quant.py` |
| Fused-epilogue GEMM | `instances/gemm_multi_d.py`, `gemm_multi_abd.py`, `helpers/fuse.py` |
| MFMA reference GEMM | `instances/mfma_gemm.py` |
| FlatMM (small decode) | `instances/flatmm.py` |
| Batched contraction (N-D) | `instances/batched_contraction.py` |

Consider:

- Does the shape fill MFMA tile dimensions? Use the
  `MfmaAtom` factory under `helpers/atoms.py` to verify the shape
  matches.
- Is M / N / K too small? Use `flatmm` or `mfma_gemm` instead of
  `gemm_universal` for tiny shapes.
- Is grouped scheduling better than padding to a common shape?
  `GroupedGemmSpec` handles this.
- Is split-K useful, or only adding atomic overhead?
- Is Stream-K needed for load balancing?
- Is weight preshuffle worth the preprocessing cost?
- Is data reused enough to justify LDS?
- Is A or B naturally contiguous? (drives `layout="RCR"` etc.)

### 4.2 Convolution

| Variant | Instance |
|---|---|
| Direct convolution | `instances/conv_direct_grouped.py::DirectConv16cSpec`, `DirectConv4cSpec` |
| Implicit GEMM | `instances/conv_implicit_gemm.py::ImplicitGemmConvSpec` |
| Implicit GEMM (auto-unrolled) | `instances/conv_implicit_gemm_auto.py` |
| im2col + GEMM | `instances/img2col.py` materializes the im2col operand |
| Winograd | not yet implemented |
| FFT | not yet implemented |
| Depthwise / small-channel | `DirectConv4cSpec` (4-channel grouped) |

Consider:

- If `K` or `C * R * S` is tiny, implicit GEMM may be structurally
  weak.
- Direct conv can preserve spatial/channel reuse.
- Small-channel conv may need custom MFMA mapping: `4x4x4`,
  `16x16x16`, `16x16x32`. The DSL ships all three (see
  `helpers/atoms.py::MFMA_F16_ATOMS`).
- Use circular row accumulators for `R` rows.
  `conv_direct_grouped.py` documents the streaming-row 3-acc circular
  pipeline at lines 41-66.
- Stream input rows so one input row contributes to multiple output
  rows.
- Keep filter weights in registers when small.
- Stage weights in LDS only if cooperative load + register pressure
  justifies it.
- Prefer direct stores unless a proven LDS epilogue improves
  coalescing enough to pay for barriers (see Section 9.3 + Empirical
  Case Study 1).

#### 4.2a The Implicit GEMM VGPR Tax

Implicit GEMM convolution pays an unavoidable "VGPR tax" compared to
pure GEMM due to im2col coordinate computation overhead. Understanding
this helps set realistic performance expectations.

VGPR Breakdown Comparison (3×3 conv, 64×128 tile):

| Component | Pure GEMM | Implicit GEMM Conv | Δ |
|---|---|---|---|
| MFMA Accumulators | 32 | 32 | 0 |
| Address Computation | 5-8 | 15-20 | +10-12 |
| LDS Offsets | 5-8 | 8-12 | +3-4 |
| Temporary Vectors | 12 | 12 | 0 |
| Loop Indices | 3-5 | 5-8 | +2-3 |
| **TOTAL** | **57-65** | **72-84** | **+15-19** |

Why the extra VGPRs?

- Computing spatial coordinates (n, ho, wo) from a linearized output
  index.
- Computing input coordinates (hi, wi) from (ho, wo, y, x) with
  stride/dilation.
- Bounds checking for padding regions.
- Filter position tracking (y, x within R×S window).
- Group offset computation for grouped convolutions.

Practical consequences:

- Implicit GEMM convolution typically achieves **60-65 % of theoretical
  peak**.
- Pure GEMM typically achieves **75-85 % of theoretical peak**.
- The 15-20 VGPR overhead reduces occupancy: 72 VGPRs → 7 waves/CU vs
  60 VGPRs → 8 waves/CU.
- This is a **fundamental algorithmic constraint**, not a tuning
  failure.

**⚠ WARNING**: Do NOT chase VGPR reduction below the im2col floor
(~60 VGPRs for 3×3 conv with typical tile sizes). The coordinate
arithmetic is mathematically required. Attempting to reduce VGPRs
further will:

- Require smaller tile sizes (reducing reuse and performance).
- Force spilling to memory (catastrophic for performance).
- Not achieve GEMM-level efficiency regardless of optimization effort.

When to accept the VGPR tax:

- Your conv kernel is within 5 % of other mature frameworks.
- You are at 60-65 % of theoretical peak for compute-bound shapes.
- Further VGPR optimization attempts have failed or regressed.

When to consider alternatives:

- Very small spatial dimensions: direct conv may have lower overhead
  (use `DirectConv16cSpec` / `DirectConv4cSpec`).
- Large filter sizes (5×5, 7×7): Winograd or FFT may amortize transform
  cost.
- Extreme channel counts with small spatial: specialized layouts.

Invest optimization time in memory access patterns, LDS swizzling, and
tile sizes rather than trying to eliminate the fundamental VGPR
overhead of im2col addressing. Use `probe_occupancy.py` to confirm the
VGPR floor before deciding.

### 4.3 Attention

| Variant | Instance |
|---|---|
| Prefill + decode (unified, scalar oracle) | `instances/attention_unified.py` |
| Prefill + decode (2D MFMA) | `instances/attention_tiled_2d.py::UnifiedAttention2DTiledSpec` |
| Split-KV decode segment + reduce | `instances/attention_tiled_3d.py` |
| FMHA forward (MFMA, prefill) | `instances/fmha_mfma.py` |
| FMHA varlen | `instances/fmha_varlen.py` |
| FMHA head grouping (GQA / MQA) | `instances/fmha_head_grouping.py` |
| FMHA paged prefill | `instances/fmha_paged_prefill.py` (warp-scalar inner) |
| FMHA splitkv decode | `instances/fmha_splitkv_decode.py` (warp-scalar inner) |
| FMHA fp8 forward | `instances/fmha_fwd_fp8.py` |
| FMHA backward | `instances/fmha_bwd.py` (warp-scalar inner) |
| FMHA append KV | `instances/fmha_appendkv.py` (DMA) |
| Sage attention (per-block scaled) | `instances/sage_attention.py` |
| Block-sparse attention (Jenga / VSA) | `instances/sparse_attention.py` |

Consider:

- Q / K / V head dimension alignment to MFMA. Use
  `helpers/attention.py::mfma_16x16x16_for_dtype` to pick a valid
  atom.
- Numerically stable max/sum accumulation via
  `helpers/attention.py::OnlineSoftmaxState` +
  `warp_xor_reduce_max`/`_sum`.
- Page-table overhead — choose between
  `attention_tiled_2d` and `attention_tiled_3d`. The
  `helpers/attention.py::select_2d_config` / `select_3d_config` and
  `use_2d_kernel` heuristics decide automatically; you can override
  with `_select_2d_num_warps`/`_select_2d_tile_size` monkey-patches
  for sweeps (see `utilities/tools/dsl_probes/probe_config_sweep.py`).
- Coalescing across heads / tokens / pages.
- Shared memory capacity for Q / K / V tiles.
- Register pressure from accumulators and softmax state. The 2D path
  carries Q in LDS once and async DMAs K/V (see Section 6.3).
- Store path for output and log-sum-exp.

### 4.4 Reductions And Normalization

| Variant | Instance |
|---|---|
| Row reduce (sum/max/min/mean/prod) | `instances/reduce.py::Reduce2DSpec` |
| LayerNorm forward | `instances/layernorm2d.py::LayerNorm2DSpec` |
| RMSNorm forward | `instances/rmsnorm2d.py::RMSNorm2DSpec` |
| Add + RMSNorm + rdquant (fused) | `instances/add_rmsnorm2d_rdquant.py` |
| Smoothquant | `instances/smoothquant.py`, `moe_smoothquant.py` |
| Topk softmax | `instances/topk_softmax.py` |
| Pooling | `instances/pooling.py::Pooling2DSpec` |

Consider:

- Reduction axis length.
- Number of rows.
- Whether one block per row is enough.
- Whether multiple CTAs per row require atomics or a second pass.
- Vector width (`sweep_row_chunks` accepts `vec`).
- Numerically stable accumulation (`helpers/reduction.py::block_lds_reduce`
  has `sum / max / min / prod`).
- Fusing scale / bias / activation via the
  `FusedEpilogue` system or directly in the kernel.

### 4.5 Transpose, Permute, Copy

| Variant | Instance |
|---|---|
| 2D transpose (LDS-staged) | `instances/transpose.py::Transpose2DSpec` |
| Batched 2D transpose | `instances/batched_transpose.py` |
| In-register sub-tile transpose | `instances/transpose_bc.py` |
| N-D permutation | `instances/permute_nd.py::PermuteSpec` |
| im2col copy | `instances/img2col.py` |

- Use vectorized global loads/stores.
- Use LDS tile for non-coalesced stores.
- Avoid bank conflicts with padding or swizzle (Section 6.4a).
- Use rectangular tiles tuned to the aspect ratio.
- Avoid over-general rank logic in hot loops; `permute_nd.py` is
  one-thread-per-element with rank-N index decompose for clarity, not
  for peak throughput.

### 4.6 Gather, Scatter, Paged, Sparse

- Index load overhead may dominate.
- Coalesce metadata loads.
- Cache page tables.
- Batch by locality.
- Avoid divergent branches.
- Use buffer descriptors with proper bounds (`buffer_rsrc`).
- Precompute or compact index maps.
- Separate dense fast path from sparse fallback.
- For MoE, see `instances/fused_moe.py`, `moe_sorting.py`,
  `moe_gemm_fused.py`, `fused_moe_e2e.py`.

---
Next: [work-decomposition](work-decomposition.md). Per-family variant levers: [knob-catalog-and-sweep](../03-autotuning/knob-catalog-and-sweep.md) (S12.1.A). Op checklists: [op-specific-checklists](../03-autotuning/op-specific-checklists.md).
