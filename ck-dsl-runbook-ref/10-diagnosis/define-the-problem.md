---
name: define-the-problem
description: Runbook S1: pin the operation contract, shapes (compile-time vs runtime), layouts, dtypes/numerics/tolerances, and boundary conditions before choosing how the kernel runs.
source: ck-dsl-optimization-runbook.md (lines 95-190)
---

## 1. Define The Problem Exactly

### 1.1 Operation Contract

Decide what the kernel computes before deciding how it should run.

- Name the operation precisely. The DSL has a separate spec dataclass
  for each kernel family: `UniversalGemmSpec` for GEMM, `ConvProblem` +
  `ImplicitGemmConvSpec` for implicit-GEMM conv, `DirectConvProblem` +
  `DirectConv16cSpec` / `DirectConv4cSpec` for direct grouped conv,
  `UnifiedAttentionProblem` plus `UnifiedAttention2DSpec` / `*3DSpec` /
  `*ReduceSpec` for paged attention, `Reduce2DSpec` / `LayerNorm2DSpec` /
  `RMSNorm2DSpec` / `ElementwiseSpec` / `Transpose2DSpec` for the
  small-op family, `FmhaVarlenSpec` / `FmhaHeadGroupingSpec` /
  `FmhaPagedPrefillSpec` / `FmhaSplitKvDecodeSpec` / `FmhaFwdFp8Spec`
  / `FmhaBwdSpec` / `FmhaAppendKvSpec` for the FMHA variants, and so
  on. The spec name is the contract.
- State whether the operation is forward, backward-data, backward-
  weight, inference-only, training, deterministic, or approximate.
  Backward FMHA lives in `FmhaBwdSpec`; everything else in
  `instances/` is forward.
- State whether the op is standalone or part of a graph/fusion
  boundary. Fusion is in `helpers/fuse.py` (`compile_fn`, `explain_fn`,
  `FusedEpilogue`, `_PATTERN_TABLE`); see `dsl_docs/fusion/overview.md`.
- State whether atomics are allowed (`b.global_atomic_add_f32`,
  emitted by split-K GEMM and Stream-K).
- State all side effects: workspace (`WorkspacePool`,
  `*_workspace_bytes`), scratch buffers, stream usage, global memory
  writes, persistent state.

### 1.2 Shapes

- Record dimensions with names, not values: every problem dataclass
  uses explicit field names (`ConvProblem.N`, `Hi`, `Wi`, `C`, `K`,
  `R`, `S`, `sH`, `sW`, `pH`, `pW`, `dH`, `dW`; `UnifiedAttentionProblem.
  total_q`, `num_seqs`, `num_query_heads`, `num_kv_heads`, `head_size`,
  `block_size`, `max_seqlen_q`, `max_seqlen_k`, …).
- State which dimensions are compile-time constants (the spec fields)
  and which are runtime (manifest fields, kernel args).
- Identify pathological boundary shapes: `1, 2, 3, 15, 16, 17, 31, 32,
  33`, powers of two, non-multiples of tile sizes. Add them to the
  parity harness in `examples/`.

### 1.3 Layouts

- Layouts are explicit in the spec: `UniversalGemmSpec.layout = "RCR"`
  / `"CRR"` / etc.; conv uses NHWC by default (see
  `dsl_docs/instances/convolution.md`); attention uses (block, slot)
  paging for KV.
- For non-trivial mappings (paged KV, im2col), the coordinate
  transform DAG lives in `transforms.py`. Operators are
  `pass_through`, `pad`, `pad_dynamic`, `embed`, `unmerge`, `merge`,
  `indirect`. Walkthrough: `architecture/TRANSFORM_DAG.md`.
- State alignment and contiguity at the descriptor: `TensorDescriptor`
  in `transforms.py` and the legacy `TensorView` / `TileWindow` in
  `helpers/tensor_view.py`.

### 1.4 Dtypes And Numerics

- Record input dtype, weight dtype, accumulator dtype, output dtype,
  scale dtype, bias dtype, index dtype. The spec dataclass owns these
  fields explicitly; e.g. `UnifiedAttention2DTiledSpec.dtype="bf16"`,
  `kv_storage_dtype="fp8e4m3"` for the FP8 KV cache path.
- DSL dtype catalog: `I1`, `I8`, `I16`, `I32`, `I64`, `BF16`, `F16`,
  `F32`, `FP8E4M3`, `BF8E5M2` (see `core/ir.py:34-97`).
- Quantization helpers in `helpers/quant.py` (`quant_max_abs`,
  `quantize_scalar_f32`, `dequantize_scalar_to_f32`, `quant_ir_type`,
  `ir_to_qdtype`). `QDType` is `Literal["i8", "fp8e4m3", "bf8e5m2"]`.
- MX block-scale support in `helpers/mx_scale.py` and the spec
  `MxGemmSpec`.
- Always set a tolerance policy. The `examples/ck_tile_parity.py`
  harness encodes the per-op tolerances we currently believe in
  (elementwise linear ops bit-exact, silu/gelu `<= 2e-4`, layer/rms
  norm `<= 5e-3` (was 2.5e-3 before noise widening, see
  `notes/PROPOSALS_IMPLEMENTATION_REPORT.md::F2`), reduce `<= 1.5e-3`,
  gemm `<= 7e-2`).
- For fp16 inputs with fp32 accumulation over O(100) terms, expect
  errors near the fp16 ULP floor for correct kernels. Errors two
  orders of magnitude higher almost always indicate structural bugs
  (wrong lane mapping, missing acc reset, stale LDS, etc.). See
  `utilities/skills/empirical-case-studies.md` (Case Study 3) for
  specific bug signatures.

### 1.5 Boundary Conditions

- Padding via `transforms.pad`, `pad_dynamic` (returns `(offset,
  valid)` per coordinate).
- Buffer descriptors with `flags=0x00027000` (= 159744) return zero
  for out-of-bounds lanes — the canonical tail-safe primitive.
  Constructed with `b.buffer_rsrc(ptr, num_bytes)`.
- Causal masks, sliding window, softcap in `helpers/attention.py`
  (`causal_mask`, `sliding_window_mask`, `apply_softcap_log2`).
- Split-K / persistent / stream-K decisions move into `streamk_gemm.py`
  + `helpers/streamk.py` + `helpers/persistent.py`.
- Page-table indirection for paged attention via
  `transforms.indirect(...) + unmerge(...)`.

---
Next: [establish-baselines](establish-baselines.md). Lane-mapping K-pack tolerance bug: [work-decomposition](../20-levers/work-decomposition.md). Spec dataclasses are the contract — see the family tables in [algorithmic-mapping](../20-levers/algorithmic-mapping.md).
