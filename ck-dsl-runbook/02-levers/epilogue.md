---
name: epilogue
description: Runbook S9: epilogue ops (bias/act/residual/scale/quant/cast), direct vs LDS/CShuffle epilogue trade-off, the 'in-between naive LDS epilogue loses both ways' caveat, and output validation. Vectorizing the epilogue is often the single largest win.
source: ck-dsl-optimization-runbook.md (lines 1179-1246)
---

## 9. Epilogue Optimization

### 9.1 Common Epilogues

- Bias add (`helpers/fuse.py::BiasAdd`).
- Activation (`ReLU`, `SiLU`, `GELU`, `tanh`, `sigmoid` —
  `helpers/fuse.py`, `EpilogueOp` chain).
- Residual add (`ResidualAdd`).
- Scaling (`Scale`).
- Quantization / dequantization (`helpers/quant.py`).
- Type conversion (`vec_trunc_f32_to_f16`, `cvt_pk_f32_fp8`).
- Row / column scale (`helpers/mx_scale.py` for MX).
- Clamp.
- Dropout (`helpers/rng.py::dropout_mask_pair_f32`).
- Store transpose (`instances/transpose*` + LDS reader formula).

### 9.2 Direct Epilogue

`helpers/epilogues.py::DirectEpilogue(vec_in_acc=True/False)`.

- Best when accumulator lane layout already matches the global store
  layout.
- Convert vector accumulator to vector output.
- Use packed stores (`b.buffer_store_vN_f16`, `b.global_store_vN_f16`).
- Avoid scalar stores.
- Avoid unnecessary LDS.

### 9.3 LDS / Shuffle Epilogue

`helpers/epilogues.py::CShuffleEpilogue.from_grid(...)`.

- Use when accumulator lane layout is bad for global stores.
- Write per-lane accumulators to LDS.
- Barrier.
- Have a subset of threads read contiguous vectors from LDS.
- Store wide vectors globally.
- Ensure LDS write and read mappings are exact.
- Ensure only active store threads write.
- Include OOB checks for Q/N/K tails.
- Measure whether coalescing gain beats barrier/LDS cost.

"Use LDS for the epilogue" is not the same as "match the library's
LDS epilogue distribution". A naive LDS-staged epilogue with a flat
`[q, group, k]` layout was correct for one shape but wrong for
another and never beat direct vector stores in our experiments. CK's
LDS epilogue uses the MFMA distribution for the LDS write, then a
separate wide-store distribution where only `STORE_VECS = BLOCK_Q ×
BLOCK_C8` threads issue 16-byte global stores. Either copy that exact
mapping or stay with direct vector stores; the in-between case loses
both ways.

A related caveat applies to any structural change that removes one
cost: **the cost may simply move**, not disappear. **§17.4** has a
worked example where removing a per-iter LDS-store stripe replaced it
with a much larger cross-lane permute storm — a material regression
on the same shape. Always confirm via the per-iter ISA histogram
(`probe_isa_inspect.py` or `probe_intrinsic_counts.py`) that the
savings outnumber the introduced overhead before declaring the
change a win.

### 9.4 Output Validation

- Epilogue bugs often affect only certain groups / channels /
  columns.
- Validate all channels, all groups, and both boundary and interior
  Q.
- Test `groups` values that change block count.
- Test both K tails and Q tails.

---
Epilogue knobs: [knob-catalog-and-sweep](../03-autotuning/knob-catalog-and-sweep.md) (S12.1.E). The cost-moved caveat: [unified-attention-2d](../05-case-studies/unified-attention-2d.md) (S17.4). ISA store-width check: [isa-resource-inspection](isa-resource-inspection.md).
