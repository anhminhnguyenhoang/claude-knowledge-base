---
name: cross-references
description: Runbook S20: pointers to the sibling runbook docs (runbook_mapping, runbook_compliance, measured_results), the utilities/skills briefs, the static probes, and the profiling-counter tool stages.
source: ck-dsl-optimization-runbook.md (lines 2841-2864)
---

## 20. Cross References

- **Short loop**: §0 "The Loop" at the top of this runbook.
- **DSL primitive map**: `runbook_mapping.md`.
- **Compliance table with measurements**: `runbook_compliance.md`.
- **Validation pass output**: `measured_results.md`.
- **Skill briefs**: `utilities/skills/*.md` —
  - `bisect-perf-regression.md`
  - `capture-kernel-trace-ckdsl.md`
  - `empirical-case-studies.md`
  - `gemm-optimization-ckdsl.md`
  - `kernel-launch-guide.md`
  - `kernel-trace-analysis.md`
  - `lds-optimization-ckdsl.md`
  - `prefetch-data-load-ckdsl.md`
- **Static inspection probes**: `utilities/tools/dsl_probes/`
  (see also `dsl_probes/README.md` for a when-to-use index).
- **Profiling-counter tools**: `utilities/tools/stage4_analyze/`,
  `utilities/tools/stage5_compare/`, `utilities/tools/utils/`.
- **Benchmark harnesses**: `utilities/tools/stage1_benchmark/`.
- **Target architecture reference**: **§21** for gfx950 / CDNA4
  MFMA atoms, LDS specs, cross-lane primitives, register / occupancy
  caps, chiplet / XCD, buffer descriptors, fp8 / MX support, and
  gfx950-specific compiler caveats.

---
These point at live paths in the upstream ck_dsl tree — verify they resolve before relying on them. Arch specifics: [target-architecture-gfx950](target-architecture-gfx950.md).
