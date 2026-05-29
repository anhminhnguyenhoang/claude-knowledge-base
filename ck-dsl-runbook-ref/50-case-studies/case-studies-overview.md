---
name: case-studies-overview
description: Runbook S17 intro + S17.1 bake-off summary (conv/direct-conv lever ladders) + S17.2 validation pass + S17.3 attention parity pass (the parity-of-order-1 = structural-bug signature).
source: ck-dsl-optimization-runbook.md (lines 2013-2115)
---

## 17. Empirical Case Studies

For concrete, measured findings from real optimization experiments,
see `utilities/skills/empirical-case-studies.md`. It contains:

- Direct-conv progression from a scalar baseline to MFMA-tiled
  parity with the reference, with the exact lever per step
  (Case Study 1).
- LDS swizzle XOR vs padding comparison with measured deltas across
  multiple architectures (Case Study 2).
- Numerical tolerance signatures and bug patterns (Case Study 3).
- Stability and measurement caveats (Case Study 4).
- Closing the last few percent — what doesn't work (Case Study 5).
- Implicit GEMM vs pure GEMM overhead (Case Study 6).

Use these as reference points to set expectations and recognize bug
signatures. They are specific to particular experiments — actual
results depend on hardware, shape, dtype, and memory hierarchy
behavior. Always benchmark on your target hardware.

### 17.1 Bake-off Summary (DSL)

For the implicit-GEMM conv bake-off
(`example/ck_tile/dsl/08_bake_off_implicit_gemm`), the canonical
DSL example of applying the runbook's levers in series:

| Lever | Direction | Notes |
|---|---|---|
| baseline (single-buffer LDS, direct epilogue, smaller-K atom) | reference | `bad=0` at conv tolerance |
| + cshuffle epilogue (§9.3) | small improvement | LDS-staged fp16 + wide stores |
| + buffer-rsrc DW3 = `0x00027000` (§6.1) | correctness fix | was producing all-zero outputs |
| + larger-K MFMA atom (§7.1) | major improvement | halves the K-loop trip count |
| + K-padded LDS (§6.3) | small improvement | breaks `ds_read` bank conflicts |
| + graph mode / amortized launch (§12) | further improvement | reduces per-launch overhead |

Each lever was empirically verified with a hygienic benchmark, all
preserving `bad = 0` correctness at the conv tolerance against the
grouped NumPy reference. The cumulative result is a multi-× speedup
over the baseline, by applying five distinct runbook levers in
series.

For the direct-conv bake-off
(`example/ck_tile/dsl/09_bake_off_direct_conv_16c`, `10_…_4c`), the
two dominant levers were:

- Switching from scalar epilogue stores to vectorised stores
  (`buffer_store_dwordx2` per lane in the 4-channel path; `dwordx4`
  in the 16-channel path) — a major improvement on both shapes.
- Folding the inner K dimension into a larger-K MFMA atom for the
  16-channel path — a smaller but stable additional improvement.

These are the same two levers (epilogue vectorisation, larger-K atom)
that dominate every MFMA-tiled kernel's optimisation log.

### 17.2 Validation Pass Results

The documented validation pass at the time of writing exercises:

- The 286-test static unit suite (`test_ck_dsl.py`) — IR construction,
  transform DAG, helpers, instance smoke.
- `verify_dsl_docs.py` — imports every symbol referenced by the docs,
  exercises every IR builder method, lowers every spec to LLVM / HIP
  / CK Tile, builds HSACO, launches small kernels.
- `test_ck_dsl_examples.py` — discovers every `example/ck_tile/dsl/`
  manifest, builds it, runs `run_manifest --verify`, asserts the
  declared tolerances.
- The bake-off 08 implicit-GEMM manifest — verifies bit-level
  correctness at the conv tolerance plus the declared TFLOPS / GB/s
  lower bounds.
- `ck_tile_parity.py --op all` — small-op parity vs torch across the
  20 cases listed in §13.3.
- `parity_extended_kernels.py --op all` — FMHA / sage / sparse / MoE
  / block-scale / MX parity vs torch / NumPy references.
- `parity_unified_attention.py` — Triton + reference vs CK DSL across
  the documented attention scenarios.

See `measured_results.md` for the latest documented validation pass
numbers; it is updated each time a major verification sweep lands.

### 17.3 Attention Parity Pass

A documented attention parity sweep covering the FMHA family
(`parity_extended_kernels --op all`), the unified-attention 2D / 3D /
auto lanes (`parity_unified_attention.py`), and the full
LLVM ↔ HIP-debug lowering audit (`hip_lowering_parity --case all`)
all passed end-to-end after two material correctness fixes landed:

- **2D ALiBi / QQ-bias correctness fix.** The transposed-32x32
  softmax block was skipping ALiBi / QQ-bias addition entirely. The
  symptom was `max_abs` *of order 1* on three scenarios while every
  other scenario stayed at the expected `fp16` ULP tolerance — a
  classic structural-bug signature (the score wasn't actually
  computed; it was missing an addend entirely). The fix wires both
  into the score computation inline before the per-row max reduce.
- **HIP-debug backend ops.** The HIP-debug lowering was missing a
  handful of vector ops (`vector.add`, `vector.mul`) and packed
  conversions (`cvt_pk_{fp8,bf8}_f32x4`). Production LLVM lowering
  was unaffected; the parity gap was purely a backend coverage gap.

Take-away: the parity-of-order-1 failure mode is structural, not
numeric noise. Always check whether a missing addend / mask term in
the score path explains an order-of-magnitude max_abs jump before
suspecting MFMA / lane-layout bugs (whose signatures are in the

---
Detailed passes: [unified-attention-2d](unified-attention-2d.md), [fused-moe](fused-moe.md), and the [skinny-decode-gemm](skinny-decode-gemm/README.md) walkthrough (full 22-step pass; runbook §17.7 condensed form: [runbook-17-7-condensed](skinny-decode-gemm/runbook-17-7-condensed.md)).
