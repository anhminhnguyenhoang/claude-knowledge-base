---
name: fused-moe
description: Runbook S17.6 active-tile dispatch + S17.5 preshuffle-B: two stacking MoE levers (ATS skips inactive CTAs; preshuffle makes per-K-tile B-load one contiguous burst), measured speedups vs CK Tile C++, and lessons (old breakdowns lie; bitwise parity bar; documented-but-unimplemented knobs).
source: ck-dsl-optimization-runbook.md (lines 2316-2553)
---

### 17.6 Fused MoE Active-Tile Dispatch

After Round 10's preshuffle work, an `(E ∈ {2,4,8,16}, topk=2)`
sweep on `decode_T1_H4096_I7168` showed e2e time scaling ~linearly
with `experts`, not with `min(experts, topk*tokens)`:

```text
E= 2  active=2  preshuf  262.2 us
E= 4  active=2  preshuf  281.0 us  (+18.8 / +9.4 per inactive)
E= 8  active=2  preshuf  363.5 us  (+82.5 / +13.7 per inactive)
E=16  active=2  preshuf  575.7 us  (+212.2 / +15.2 per inactive)
```

CK Tile avoids this waste with an active-tile dispatcher: each CTA
reads `sorted_expert_ids[block_id_z]` to pick its expert and
returns early when the sorted-tile id exceeds `num_sorted_tiles`.
Round 11 ports the idea into the DSL kernels.

**Implementation surface.**

1. `instances/gemm_universal.py::TraitSpec.active_tile_skip`: new
   field, default `False`. `kernel_name` picks up an `actt` flag.
2. `build_universal_gemm`: when `batched and active_tile_skip`,
   declare two extra params (`SortedTokenIds: ptr<i32>`,
   `slot_size: i32`). Compute
   `bucket_head = block_id_z * slot_size + block_m_off`,
   `do_work = SortedTokenIds[bucket_head] >= 0`. Wrap the K-loop +
   epilogue in `scf.if(do_work)`. Using `block_m_off` (already
   chiplet-swizzle-aware, tile_m-aligned) keeps the gate
   consistent with the address arithmetic the body actually uses.
3. `build_moe_interleaved_gate_up_silu_gemm`: same gate, same
   `block_m_off` form.
4. `instances/batched_gemm.py::batched_gemm_signature` and
   `moe_interleaved_gate_up_silu_gemm_signature`: append the two
   extra args when `trait.active_tile_skip`.
5. `instances/fused_moe_e2e.py::FusedMoeForwardSpec.active_tile_skip_gemms`:
   orchestrator-side knob. A parameterized launcher cache
   (`_moe_batched_gemm_launcher`,
   `_moe_interleaved_gate_up_silu_launcher`) returns the right
   HSACO for any combination of `(preshuffle_b, active_tile_skip)`
   on demand, so the four binary variants per kernel family share
   one cache.
6. `examples/moe/test_active_tile_skip.py`: standalone parity +
   perf harness. Confirms bitwise-equal output to baseline when
   all tiles active, zero output for inactive tiles, ~14× speedup
   when every tile is inactive.

**Measured outcome.**

Standalone kernel (B=8, M=32, N=4096, K=4096, fp16):

```text
base no skip       172.31 us
att, all-active    173.85 us   (0% overhead)
att, all-inactive   13.10 us   (13× faster — kernel just exits)
```

End-to-end MoE (HIP-graph-replayed, fp16):

```text
                                baseline   preshuf_intl   +ATS    vs base
decode_T1_E8  H=4096 I=4096     269.4 us   227.5 us       166.3   1.62×
decode_T1_E8  H=4096 I=7168     412.7 us   362.8 us       264.2   1.56×
decode_T8_E8  H=4096 I=4096     488.4 us   229.1 us       227.5   2.15×
decode_T8_E8  H=4096 I=7168     382.5 us   365.0 us       351.8   1.09×
decode_T1_E16 H=4096 I=7168     620.1 us   544.9 us       264.0   2.35×
```

vs CK Tile C++ on the canonical decode (router in DSL timing):

```text
decode_T1_E8_K2_H4096_I7168 :
    ck_dsl baseline     0.379 ms  (0.32× of cktile)
    ck_dsl preshuf      0.357 ms  (0.34× of cktile)
    ck_dsl preshuf+ATS  0.265 ms  (0.46× of cktile)
    ck_tile_cpp         0.121 ms
```

Closed roughly 30 % of the gap to CK Tile on the canonical
decode_T1, and ~50 % on `decode_T1_E16`. Bitwise parity
(`rel=0.00e+00`) across every standalone and e2e variant in
`test_active_tile_skip.py`, `test_fused_moe_preshuffle.py`,
`test_preshuffle_b.py`.

**Lessons reinforced.**

* **Old breakdowns lie.** The runbook §17.5 / Round 1 attribution
  ("`fused_moe_reduce` is 11 % / 46 µs") was stale. Clean
  HIP-graph timing measured the reduce kernel at ~9 µs alone and
  ~3 µs in chain. After preshuffle, the GEMM kernels (gate_up +
  down) are ~99 % of the per-replay time. Always re-measure the
  hot-kernel attribution after a meaningful structural change.
* **Launch overhead is not the problem on these shapes.** Eager
  forward and HIP-graph replay are within 2 µs of each other
  (407 µs vs 410 µs baseline; 351 µs vs 356 µs preshuf). Mega-kernel
  fusion will not help because of fewer launches; it helps
  because of fewer HBM round-trips and shared register state.
* **Active-tile dispatch is independent of preshuffle and stacks
  with it.** Both are valid levers for different reasons:
  preshuffle reduces per-K-tile B-load time; ATS reduces the
  number of CTAs doing useful work. They compose multiplicatively
  on the right shapes (`decode_T1_E8`: 1.62× combined vs ~1.07×
  preshuf alone).
* **`block_m_off` not `block_id_y * tile_m`.** When a kernel does
  any tile-id remap (chiplet swizzle, persistent grid, etc.) the
  gate must read the bucket head for the *post-remap* row, or it
  will skip a tile the body still tries to compute (or vice
  versa). Use whatever value the kernel uses to address A and C.

---

### 17.5 Fused MoE Preshuffle-B Implementation

The fused-MoE optimization pass (`examples/moe/`) ran 9 rounds of
config-level tuning before stalling at the configuration ceiling.
Round 9 named `preshuffle_b=True` as the genuinely-untried lever
from §12.1.H but discovered the flag was a documented-but-silently-
ignored knob: declared in `gemm_universal.py::TraitSpec` but never
read by `build_universal_gemm()`. Round 10 closes that gap.

**The problem.** With the canonical row-major B layout
`(N, K)`, each per-K-tile B-load in the MFMA inner loop visits
`block_n` rows of B at offsets `K` apart. A wave's lanes hit
different rows; the GPU coalescer can only batch a partial wave
into a single `buffer_load_dwordx4`, so each K-step costs multiple
discontiguous VMEM transactions.

**The transform.** Pre-shuffle B once on the host into
`(E, k_tiles, n_tiles, block_n, block_k)` contiguous, where
`k_tiles = K / block_k` and `n_tiles = N / block_n`. The
`(block_n × block_k)` tile that the per-K-tile B-load wants is now
exactly `block_n × block_k × elem_bytes` consecutive bytes. One
wide contiguous burst per warp replaces the strided per-row loads.
The in-tile element order matches `B_smem`'s row-major
`(block_n, block_k)` layout, so the MFMA inner loop's `ds_read`
pattern is unchanged.

**Implementation surface.**

1. `gemm_universal.py::emit_load_phase`: add a
   `if spec.trait.preshuffle_b:` branch that computes
   `tile_offset = (k_tile * n_tile_count + n_tile) * (block_n *
   block_k)` and issues
   `b.global_load_vN(B, base_off + vec_idx * load_vec, dtype,
   load_vec)` directly (bypassing `TileWindow` which models the
   strided 3-D view).
2. `moe_gemm_fused.py::build_moe_interleaved_gate_up_silu_gemm`:
   same branch; the only delta vs `gemm_universal` is the
   `n_tile_count = (2*N) / block_n` (the GEMM N is `2*N` because
   gate and up are packed along the N axis).
3. `gemm_universal.py::UniversalGemmSpec.kernel_name`: append a
   `preb` flag so HSACO caches don't alias.
4. `fused_moe_e2e.py`: add three orchestrator knobs
   (`preshuffle_w_down`, `preshuffle_w_gate_up_packed`,
   `preshuffle_w_gate_up_interleaved`), three host-side
   preshuffle helpers with data-ptr-keyed caches, and pre-build
   the preshuffled tensors inside `capture_graph` *before* the
   warmup loop so the one-time `torch.cat / permute / contiguous`
   cost stays out of the captured / replayed region.
5. `examples/moe/test_preshuffle_b.py`: standalone batched-GEMM
   parity + perf harness that flips `preshuffle_b` and confirms
   bitwise-equal kernel outputs (`max|delta|=0.0` between the
   two paths) plus per-shape timing.
6. `examples/moe/test_fused_moe_preshuffle.py`: end-to-end harness
   that runs five FusedMoeForward configurations
   (`baseline_interleaved`, `baseline_packed`, `preshuf_down_only`,
   `preshuf_packed_full`, `preshuf_intl_full`) per scenario with
   subprocess-style `_isolate_lane()` between configs, and
   confirms parity-with-baseline (`rel=0.00e+00`) for every
   variant.

**Measured outcome.**

Standalone batched-GEMM at the production tile (32 × 128 × 64,
fp16): 1.5–2.1× speedup across the canonical decode shapes. Some
sample points:

```text
B=8  M=512  N=2048 K=4096  : 232.35 → 151.03 us  (1.54×)
B=16 M=256  N=4096 K=4096  : 613.14 → 318.12 us  (1.93×)
B=32 M=128  N=4096 K=4096  : 670.52 → 346.02 us  (1.94×)
B=8  M=128  N=2048 K=4096  : 222.69 → 105.26 us  (2.12×)
```

End-to-end (HIP-graph-replayed) on real MoE shapes (E=8 K=2,
fp16, with all three preshuffle knobs enabled at the optimum):

```text
                                  baseline    preshuf_intl   speedup
decode_T8 H=4096 I=4096           274.14 us   231.73 us      1.18×
decode_T1 H=4096 I=4096           267.98 us   226.44 us      1.18×
decode_T1 H=4096 I=7168           384.41 us   360.77 us      1.07×
decode_T8 H=4096 I=7168           385.05 us   367.78 us      1.05×
```

vs CK Tile C++ on the production-canonical decode scenario:

```text
decode_T1_E8_K2_H4096_I7168 :
    baseline 0.388 ms (3.20× of cktile)
    preshuf  0.355 ms (2.93× of cktile)
    cktile   0.121 ms
```

Closed ~7 % of the gap to CK Tile on the canonical shape. The
larger 18 % wins on the I=4096 shapes show the lever pays off in
proportion to how much per-K-tile B-load time the kernel was
spending: I=7168 has flatter K-axis utilization (more N tiles per
CTA), so the in-tile B-load is a smaller fraction of kernel time.

**Lessons reinforced for §17.6.**

* **Documented-but-unimplemented knobs are real lurkers.** `preshuffle_b`
  was a `TraitSpec` field for many releases; nothing in the build
  path read it. The first round of optimization noticed
  `preshuffle_b=True` was a no-op via HSACO byte-equality and
  IR-line equality. That is exactly the §17.4 "diff the lowered IR
  to verify a flag is honored" guidance applied to a config knob.
* **Bitwise parity is the right correctness bar.** All five
  end-to-end variants produced identical Y tensors
  (`max|Y_pre - Y_base| = 0`). This rules out a subtle layout-
  mismatch bug that an `atol`-based tolerance would have hidden.
* **Shape-dependent wins, again.** The I=4096 case sees 18 %; the
  I=7168 case sees 5–7 %. The kernel-level speedup is real on both
  but the share of e2e time spent in the GEMM B-load differs,
  which is the §17.4 corollary "a change that wins on 4-8 shapes
  may regress on the full production-shape distribution" applied
  to a *non-regressing* knob: the parity is preserved and the
  speedup magnitude varies, so the knob is a per-scenario lever in
  the §15 dispatcher rather than an unconditional default.
* **Mega-kernel fusion is still the next big lever.** With
  preshuffle_b in place across both GEMM bodies (universal +
  interleaved gate-up), the remaining 3× to CK Tile is the same
  structural gap: gate / up / SiLU / down / weighted-reduce
  living in one kernel with shared register state and an
  in-kernel grouped-GEMM dispatcher. That is multi-week kernel
  authoring, not a session-scoped pass — captured here as the
  explicit follow-up.

---
Preshuffle / persistent knobs: [knob-catalog-and-sweep](../30-autotuning/knob-catalog-and-sweep.md) (S12.1.H/I). MoE instances: [algorithmic-mapping](../20-levers/algorithmic-mapping.md) (S4.6). Verify-flag-is-honored discipline: [isa-resource-inspection](../20-levers/isa-resource-inspection.md).
