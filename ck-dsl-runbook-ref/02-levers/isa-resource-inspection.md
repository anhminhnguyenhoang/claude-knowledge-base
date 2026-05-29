---
name: isa-resource-inspection
description: Runbook S11: the ck_dsl.analysis layer (analyze_llvm_ir / analyze_hsaco / VariantReport), what to count, what to extract (VGPR/SGPR/LDS/spill/occupancy), and how to interpret — with a worked intrinsic-count diff.
source: ck-dsl-optimization-runbook.md (lines 1331-1406)
---

## 11. ISA And Resource Inspection

The DSL's static analysis layer lives at `ck_dsl.analysis`:

| Tool | Source |
|---|---|
| `analyze_llvm_ir(text)` | `analysis/ir.py` — counts MFMA / async / raw-buffer / barriers / waitcnts in LLVM IR |
| `analyze_hsaco(hsaco_path)` | `analysis/isa.py` — uses `llvm-objdump -d` + `llvm-readelf --notes`, returns `IsaStats` + `ResourceInfo` |
| `parse_isa(text)` | `analysis/isa.py` — opcode tally |
| `parse_resources(text)` | `analysis/isa.py` — VGPR / SGPR / AGPR / LDS / spill |
| `VariantReport.from_artifact(...)` | `analysis/report.py` — joins compile + IR + ISA + bench |
| `compare_variant_reports([...])` | sorted comparison rows |

The DSL probes under `utilities/tools/dsl_probes/` are the
**runbook-aligned interactive** surface on top of this:

```text
probe_intrinsic_counts.py   → analyze_llvm_ir-like, with custom intrinsic table
probe_isa_inspect.py        → analyze_hsaco-like, with VALU/SALU sub-buckets
probe_occupancy.py          → readelf notes + occupancy estimate
```

### 11.1 What To Count

- MFMA / WMMA instructions.
- Global loads / stores.
- LDS reads / writes.
- Barriers.
- Waits.
- Scalar ALU instructions.
- Vector ALU instructions.
- Branches.
- Atomics.
- Conversion instructions (`v_cvt_*`, `cvt_pk_*`).

### 11.2 What To Extract

- VGPR count.
- SGPR count.
- LDS bytes.
- Scratch / spill usage.
- Occupancy metadata.
- Code object target.
- Workgroup size.
- Waves per workgroup.

### 11.3 How To Interpret

- High VGPR with low occupancy may need fewer accumulators or a
  smaller tile.
- High barriers with little compute per phase suggests bigger
  per-barrier work or fewer phases.
- Many scalar stores indicate an epilogue problem.
- Many address instructions indicate offset math should be hoisted /
  precomputed.
- Register spills usually invalidate performance conclusions until
  fixed (`probe_occupancy.py` flags `spill > 0`).
- If MFMA count is higher than expected, inspect loop unrolling and
  instruction choice.

Example diff (from running `probe_intrinsic_counts.py` on the
`use_mfma_32x32` lever in `UnifiedAttention2DTiledSpec`):

```text
baseline_mw16 → mfma32_transposed:
  mfma.f32.16x16x32.bf16:   17 → 0   (-17, replaced)
  mfma.f32.32x32x16.bf16:    0 → 17  (+17)
  ds.swizzle:               32 → 0   (-32, replaced)
  ds.bpermute:               0 → 67  (+67)
  fmul (LLVM IR):           53 → 113 (+60)
```

The diff makes the algorithmic change visible: the `transposed`
variant swaps an `ds.swizzle`-based softmax butterfly for an
`ds.bpermute`-based one. The next question — "is the bpermute path
faster on this shape?" — is for `probe_targeted_bench.py`.

---
Interactive probes on top of this: [dsl-probe-workflow](../01-diagnosis/dsl-probe-workflow.md). Probe output fields: [knob-catalog-and-sweep](../03-autotuning/knob-catalog-and-sweep.md) (S12.1.Q).
