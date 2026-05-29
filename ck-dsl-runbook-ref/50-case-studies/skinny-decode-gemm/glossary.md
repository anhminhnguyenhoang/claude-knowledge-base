---
name: glossary
description: Tensile kernel-name tokens (MT/MI/DTLA/SK/PGR/PLR/SIA/WG) and CK DSL TraitSpec knobs decoded, plus the shape and hardware terms used throughout the walkthrough.
source: ck-dsl-gemm-skinny-decode-README.md (definitions consolidated from the step tables, esp. lines 307-333, 484-491)
---

# Glossary

Terms are defined as the source uses them (Qwen3-8B `o_proj` decode GEMM,
`bf16 M=2 N=4096 K=4096`, MI355X / gfx950 / CDNA4). Tile notation in the
scripts is `t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}_a{mfma_atom}_{pipeline}`.

## The shape & hardware

| Term | Meaning |
|---|---|
| **skinny / decode GEMM** | `M ≤ 8` matmul from autoregressive decode; HBM-bound, not compute-bound. Here `M=2`. |
| **`o_proj`** | The attention output projection in a transformer block; the worst-utilization GEMM in the Qwen3-8B decode trace (29% of HBM peak under rocBLAS). |
| **MI355X / gfx950 / CDNA4** | The target part: 8 TB/s HBM peak, 2.5 PFLOPS bf16 MFMA peak, 8 XCDs (chiplets). |
| **XCD** | Accelerator complex die — a chiplet with its own L2 slice; MI355X has 8. The unit `chiplet_swizzle` steers WGs onto. |
| **WG / CTA** | Workgroup / cooperative-thread-array — the threadblock launched per macro tile. |
| **HBM ceiling** | The sustained streaming-bf16 bandwidth wall: ~3.5 TB/s ≈ 44% of the 8 TB/s peak. Both DSL and rocBLAS live here. |

## Tensile kernel-name tokens (from the rocprof name)

The real hipBLASLt kernel for this shape is
`Cijk_..._MT16x16x512_MI16x16x1_DTLA1_DTLB1_GRVWA8_GRVWB8_GSU0_PGR2_PLR1_SIA3_SK3_WG16_4_4`.

| Token | Meaning | CK DSL equivalent |
|---|---|---|
| **MT 16×16×512** | macro tile `(tile_m, tile_n, tile_k)` | `tile_m=16, tile_n=16, tile_k=512` |
| **MI 16×16×1** | MFMA atom | `mfma_f32_16x16x32_bf16` |
| **GRVWA8 / GRVWB8** | global read vector-width 8 on A / B | DSL emits `dwordx8` bursts already |
| **DTLA1 / DTLB1** | DirectToLDS for A / B (load straight into LDS, bypass VGPR) | `TraitSpec.direct_to_lds` (patched in) |
| **GSU0** | Global-Split-U = 0, i.e. no split-K | not exposed in DSL |
| **SK3** | StreamK basic (no split-K, since GSU0) | not exposed in DSL |
| **PGR2 / PLR1** | 2 prefetch-global-read, 1 prefetch-local-read | `mem` pipeline (implicit-1) / `compv4` (2) |
| **SIA3** | ScheduleIterAlg=3 — explicit fixed loop-iteration schedule | implicit per pipeline |
| **WG 16,4,4** | 256-thread workgroup, 4×4 wave grid | analogous to `warp_m × warp_n` |

## CK DSL knobs (TraitSpec / pipeline)

| Knob | Meaning |
|---|---|
| **`mem` pipeline** | Memory-oriented pipeline, implicit single prefetch; the baseline winner here. |
| **`compv3` / `compv4`** | Compute-pipeline variants that trade LDS for latency hiding (compv4 ≈ PLR2). Neutral on this no-compute-to-hide shape. |
| **`direct_to_lds` (DTLA)** | Patched `TraitSpec` bool: issue `async_buffer_load_lds_addr` for A & B, removing the `global_load → VGPR → ds_write` round-trip (−32 VGPR, −32 inst / K-tile). Regresses alone; unlocks `tile_k=1024`. |
| **`dtl_prefetch`** | Double-buffered ping-pong DTLA (scf.for iter-arg carries the live half). Bit-exact, perf-neutral on this HBM-bound shape. |
| **`preshuffle_b`** | Replace strided B loads with contiguous `buffer_load_dwordxN` bursts (host permutes B). ~1.5% here. |
| **`persistent`** | Persistent-CTA grid (amortize launch cost). Noise here — only 64 macro tiles. |
| **`chiplet_swizzle`** | Remap WGIDs (via `chiplet_aware_super_tile_dynamic`) so consecutive WGs land on the same XCD, reusing A's L1 footprint. Final 2.6% lift at `wgm=8 chunk=16`. |
| **`waves_per_eu`** | Force a VGPR budget to target more waves/EU. Regression here — `MAX_WAVES_PER_CU` already the limiter. |
| **CACHE_ALL / CACHE_STREAM / NT / GLC** | Cache hints per operand. `CACHE_ALL` wins universally for both A and B; streaming / non-temporal hints regress. |
| **K-pack** | Deepen `tile_k` via the same MFMA atom to halve the K-loop trip count (§12.1.C). The 25% step-2 win. |
| **split-K (GSU / streamk)** | Tile K into `k_batch` slabs across more CTAs + a second-stage reduce. The named-but-unaddressed structural lever; doesn't help *this* shape (would replicate B traffic). |

Sources:
- Upstream `gemm_perf_skinny_decode/README.md` (the walkthrough these notes derive from).
- Token meanings cross-checked against Tensile/hipBLASLt kernel-naming conventions; verify against the live rocprof output for your build.

---
Topic index: [README](README.md). Method: [runbook-discipline](00-method/runbook-discipline.md).
