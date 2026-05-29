---
name: matrix-instructions
description: Runbook S7: MFMA/WMMA atom selection, operand layout (lane-layout matching across chained atoms — the 32x32 C-out matches A-in, 16x16 does not), accumulator strategy, and reducing MFMA count via K-pack.
source: ck-dsl-optimization-runbook.md (lines 1023-1102)
---

## 7. Matrix Instructions And Compute

### 7.1 MFMA/WMMA Selection

The DSL's `MfmaAtom` catalog (`helpers/atoms.py`) ships:

```text
f16: 16x16x16, 16x16x32, 32x32x8, 32x32x16, 4x4x4
bf16: 16x16x16, 16x16x32, 32x32x16
fp8 / bf8: native CDNA4 fp8 / bf8 MFMA + scaled / MX variants
```

See **§21.1** for the full per-arch atom catalog, including CDNA4-
only atoms (fp8 / bf8 / scaled / MX) and the lane-layout-match
property of the 32×32 atoms.

- `16x16x16` for fp16/bf16 standard tiles.
- `16x16x32` for fp16 on newer AMD architectures when K packing is
  valid.
- `4x4x4` for many independent small-channel computations
  (used by `DirectConv4cSpec`).
- Scaled MFMA for fp8 / fp6 / fp4 where available.
- WMMA for RDNA / wave32 architectures — not currently exposed.

Always confirm the emitted intrinsic with `probe_intrinsic_counts.py`.
For example, switching from the default to `mfma_f32_32x32x16_f16` is
visible immediately: the `mfma.f32.32x32x16.f16` line goes from 0 to
N, and the `mfma.f32.16x16x32.f16` line drops correspondingly.

### 7.2 Operand Layout

- Verify lane mapping with `MfmaAtom.lane_to_output(b, lane, i)`.
- Verify packed vector element order.
- Verify A/B orientation.
- Verify K packing.
- Verify output accumulator lane mapping.
- Write a tiny kernel or test if unsure.
- Compare against reference before optimizing performance.

For `mfma_f32_16x16x32_f16` on AMD CDNA, lane `(c4 = lane / 16)` holds
K elements `[c4*8 : c4*8 + 8]` (not a flat concatenation of two
4-element halves) — see Section 5.3 for the canonical bug signature.

Lane-layout matching between chained atoms matters as much as atom
shape. The `mfma_f32_16x16x32` atom's C-output lane layout does not
match its A-input lane layout, so a QK→PV chain using two 16×16 atoms
needs an LDS round-trip or `~288` cross-lane permute ops per iter to
re-pack between them. The `mfma_f32_32x32x16` atom's C-output natively
matches the A-input, so the same chain runs with zero re-pack. **§17.4**
quantifies this for unified attention: the 32×32 atom (`use_mfma_32x32`
+ `use_transposed_qk_32x32`) is structurally cheaper *because of* this
layout match, independent of the atom's compute throughput.

### 7.3 Accumulator Strategy

- Use enough accumulators to hide latency but not so many that VGPR
  pressure collapses occupancy.
- Circular accumulators can avoid shifting data.
- Reset accumulator slots at the correct lifecycle point.
- For convolution, reset phantom-row accumulators even when the
  output row is OOB. The CK direct-conv "unconditional slot reset"
  trick is non-obvious and easy to get wrong. Symptom of the bug:
  error roughly proportional to one filter coefficient times one row
  of input (we measured `max_abs ~ 1e-2`, mean ~ 1.4e-3, with ~50 %
  of elements above `1e-3`).
- For split-K, define accumulation and reduction strategy.
- For attention, keep max / sum / output accumulators consistent.
  See `helpers/attention.py::OnlineSoftmaxState`.

### 7.4 Reducing MFMA Count

- Fold small dimensions into K if operand layout supports it.
- Use larger K MFMA variants when valid (`16x16x32` over `16x16x16`,
  `32x32x16` over `32x32x8`).
- Fuse filter positions into K for convolution when contiguous.
- Use Toeplitz-like packing for convolution only when correct and
  worth the complexity.
- Avoid K32/K64 if it changes the result beyond tolerance.
- Check whether MFMA reduction count reduction causes memory / layout
  overhead elsewhere.

---
Full atom catalog: [target-architecture-gfx950](../60-reference/target-architecture-gfx950.md) (S21.1). Atom/K-pack knobs: [knob-catalog-and-sweep](../30-autotuning/knob-catalog-and-sweep.md) (S12.1.C). Quantified chain re-pack saving: [unified-attention-2d](../50-case-studies/unified-attention-2d.md) (S17.4).
