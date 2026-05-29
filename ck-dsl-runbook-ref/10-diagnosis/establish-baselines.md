---
name: establish-baselines
description: Runbook S2: layered correctness gates (cheapest-first), vendor/library perf baselines, benchmark hygiene (HIP events, median+spread, bimodal/cold-cache caveats), and metadata to record. Never report speed without correctness.
source: ck-dsl-optimization-runbook.md (lines 194-293)
---

## 2. Establish Baselines

### 2.1 Correctness Baselines

The DSL ships several layered correctness gates. Use them in order
from cheapest to most expensive:

1. `pytest test/test_ck_dsl.py` — 286 static tests (IR construction,
   transform DAG, helpers, instance smoke). ~1.7 s.
2. `python python/ck_dsl/dsl_docs/development/verify_dsl_docs.py` —
   imports every symbol, exercises every IR builder method, lowers
   every spec to LLVM/HIP/CK Tile, builds HSACO, launches small
   kernels (50 checks, ~2 s).
3. `python -m ck_dsl.run_manifest <hsaco> <manifest>.json --verify` —
   per-kernel verification: loads HSACO, builds inputs, runs the
   kernel, compares against the in-process NumPy/torch reference.
4. `python python/ck_dsl/examples/ck_tile_parity.py --op all` — small
   ops vs torch reference (20 cases, deterministic with seed=0).
5. `python python/ck_dsl/examples/parity_extended_kernels.py --op all`
   — FMHA / Sparse / Sage / MoE / Block-scale / MX correctness.
6. `python python/ck_dsl/examples/attention/parity_unified_attention.py`
   — attention parity (Triton + ref vs CK DSL paths).
7. `python python/ck_dsl/examples/hip_lowering_parity.py` — production
   LLVM lowering vs HIP-debug lowering audit across every shipped
   spec.

References do not have to be torch. The conv and GEMM bake-offs use
NumPy fp32 accumulation in `run_manifest.py`. Attention has a
deliberate per-shape `ref_paged_attn` in
`examples/attention/parity_unified_attention.py`.

### 2.2 Performance Baselines

Use the best available vendor / library baseline:

- Conv / GEMM bake-offs: CK Tile C++ via
  `lower_kernel_to_cktile`. Currently only `UniversalGemmSpec` and
  `ImplicitGemmConvSpec` lower cleanly to CK Tile C++; the rest raise
  `NotImplementedError`. See `core/lower_cktile.py`.
- Triton: AITER ships the production `unified_attention` kernel that
  vLLM and AITER use. Path comes in via `AITER_PATH` env var.
- AITER FA / FA2 for the FMHA shapes.
- Torch eager: `examples/ck_tile_parity.py::_bench_torch`.
- A naive scalar `_fmha_warp_body.py` (`fmha_warp_fwd_inner_body`) is
  the **correctness oracle** for the FMHA family. Several FMHA specs
  still ship with the warp-scalar body (paged_prefill, splitkv_decode,
  bwd, varlen on bf16 pre-F1, fp8 on the QK/PV paths) — these are
  intentionally NOT perf paths. The MFMA-tiled body
  (`mfma_attention_fwd_inner_body` in `helpers/mfma_attention.py`)
  replaces the scalar one in the production path.

Hard rule: **do not** report speed without correctness. The runbook's
top-level constraint applies verbatim here.

### 2.3 Benchmark Hygiene

- Use HIP events via `runtime/launcher.py::time_launches`. It owns
  warmup, the per-launch synchronize, and the iteration loop.
- For Triton interop, prefer the direct `torch.cuda.Event` window in
  `utilities/tools/dsl_probes/probe_targeted_bench.py::time_cuda_event`
  — `time_launches` calls `synchronize()` between iterations which
  perturbs Triton's autotuner.
- Run at least 3-5 invocations and report median + spread (defined as
  `(max - min) / median * 100 %`). The DSL benchmark summary owns
  this in `benchmark/summary.py::summarize_runs`.
- Some kernels are bimodal across runs even with identical code: a
  fresh process may hit one tier, steady-state runs settle to another.
  Always run at least 3-5 invocations and record both median and
  spread.
- Cold-cache vs warm-cache effects can produce single-run drops of
  more than 2× the steady-state throughput on the first run of a
  fresh process. Discard the first run unless you are explicitly
  measuring cold start.
- For multi-kernel pipelines, use `KernelLauncher` (one HSACO loaded
  once) or `PipelineLauncher` (multi-stage). The launcher amortizes
  the HIP module load.
- For graph replay timing, the launcher's `time_launches` accepts a
  `no_fence` context manager that elides per-call sync inside the
  event-timed loop.
- Pin GPU clocks if you suspect throttling:
  `sudo rocm-smi --setperflevel high && sudo rocm-smi --setsclk 7`.

### 2.4 Metadata To Record

The manifest writer (`helpers/manifest.py::make_*_manifest` →
`write_artifact`) records most of this automatically:

- GPU model and architecture (default ISA target is `amdgcn-amd-amdhsa
  --gfx950`).
- ROCm/CUDA version, driver/runtime version.
- Library commit SHA (set via `manifest.notes`).
- Kernel source commit SHA.
- Compile options (`compile_kernel` accepts `options`; default `-O3`).
- Target architecture flag.
- Build type and `NDEBUG`.
- Optimization flags.
- Environment variables.
- Binary code object metadata via `analyze_hsaco(hsaco_path).resources`.
- `timings` dict on the artifact records `ir_build`, `ir_lower_llvm`,
  `comgr_bc`, `reloc`, `exe`, `total`.

---
Next: [bottleneck-classification](bottleneck-classification.md). Exact commands: [reproducible-commands](../00-method/reproducible-commands.md). Hygiene knobs: [knob-catalog-and-sweep](../30-autotuning/knob-catalog-and-sweep.md) (S12.1.P).
