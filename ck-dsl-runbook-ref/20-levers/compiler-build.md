---
name: compiler-build
description: Runbook S10: build type, common AMD/HIP flags, DSL-specific compiler hazards (conservative pass pipeline; -enable-post-misched=0 miscompiles MLIR kernels; AST-rewriter if-elision trap), and alias/pointer semantics. Flags rarely close >1% gaps.
source: ck-dsl-optimization-runbook.md (lines 1250-1327)
---

## 10. Compiler And Build Settings

### 10.1 Build Type

- Use Release builds for performance.
- Ensure `NDEBUG` is set for C/C++ kernels when assertions affect
  device code (the DSL passes through to `runtime/comgr.py`).
- Confirm target architecture explicitly
  (`compile_kernel(kdef, isa="amdgcn-amd-amdhsa--gfx950")`).
- Save compile commands and binary metadata.
- Avoid comparing debug and release builds.

### 10.2 Common AMD/HIP Flags

The default flag set in `runtime/comgr.py` is `-O3`. Per-spec
overrides via `compile_kernel(kdef, options=[...])`.

- `--offload-arch=gfx950` (or `gfx942`, `gfx90a`).
- `-O3`.
- `-DNDEBUG`.
- `-fno-offload-uniform-block` when required for launch assumptions
  or perf.
- `-mllvm -amdgpu-function-calls=false`.
- `-mllvm -amdgpu-early-inline-all=true`.
- `-mllvm --lsr-drop-solution=1`.
- `-mllvm -enable-post-misched=0` only if correctness is verified
  (this flag is **known** to miscompile some MLIR-generated kernels
  even when it works for hand-written HIP — see Section 10.3).

Compiler flags very rarely close large performance gaps. Rebuilding
a kernel with the full recommended flag stack including `-DNDEBUG`,
`-fno-offload-uniform-block`, `-mllvm -amdgpu-function-calls=false`,
`-mllvm -amdgpu-early-inline-all=true`, `-mllvm --lsr-drop-solution=1`,
`-mllvm -enable-post-misched=0` has been measured to move throughput
by under 1 % across multiple kernels and DSLs. If you see a multi-×
gap, do not blame compiler flags first; instrument the ISA, the
bottleneck class, and the kernel structure (§3, §11).

### 10.3 DSL-Specific Compiler Hazards

- The DSL ships a conservative pass pipeline:
  `core/passes.py::optimize_kernel` runs canonicalize → conservative
  integer constant fold (add/sub/mul/div/mod/and/or/zext/sext/cmp/
  select) → CSE on pure ops → DCE on unused pure ops. Up to 3
  iterations. No vectorizer, no fuser, no scalar-elision pass. Loads /
  stores / MFMA / barriers are never moved.
- `core/lower_llvm.py` does not vectorize, fuse, or elide scalar
  work. The only scalar-op optimization it applies is
  `_lower_unrolled_for` (Phase 3 unroll + Phase 4a trailing-sync
  elide).
- Specific MLIR/FlyDSL trap (not our DSL, but a related one):
  `-mllvm -enable-post-misched=0` is safe in CK's CMake-generated code
  but produces wrong outputs in MLIR-generated kernels for some
  shapes. The other CK-style flags
  (`amdgpu-function-calls=false`, `amdgpu-early-inline-all=true`,
  `lsr-drop-solution=1`, `-fno-offload-uniform-block`, `-O3`, fast/
  unsafe math) were safe. Always treat compiler flags as
  one-at-a-time experiments with a correctness check after each one.
- DSL AST-rewriter trap: in tracing-style DSLs (like FlyDSL) the AST
  rewriter classifies any `if` whose test contains an `ast.Name` as
  potentially dynamic and converts it into an `scf.if` dispatch. A
  `if loads_next is not None:` guard around a Python-time decision
  is silently rewritten and the body elided, dropping all in-loop
  LDS stores while still emitting all loads, MFMAs, and barriers (and
  producing wrong but plausible-looking outputs). The CK DSL avoids
  this by raising `TypeError` on `Value.__bool__`; you MUST use
  `IRBuilder.static_if(...)` for Python booleans and
  `IRBuilder.scf_if(...)` for runtime predicates.

### 10.4 Alias And Pointer Semantics

- Preserve `noalias` metadata when safe — the DSL `IRBuilder.param`
  accepts `noalias=True`, `readonly=True`, `writeonly=True`, `align=N`,
  `dereferenceable=N`.
- Do not disable alias analysis casually.
- Mark pointers restrict / noalias in C++ when correct.
- Use buffer descriptors with proper bounds when possible.
- Avoid hidden aliasing between input / output / workspace.

---
Flag safety table: [knob-catalog-and-sweep](../30-autotuning/knob-catalog-and-sweep.md) (S12.1.M). gfx950 compiler caveats: [target-architecture-gfx950](../60-reference/target-architecture-gfx950.md) (S21.8). Anti-pattern (flags before ISA): [decision-heuristics](../40-failure-reporting/decision-heuristics.md).
