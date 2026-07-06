---
name: optimize-mla-reduce-flydsl
description: Autonomous measured loop to optimize the FlyDSL mla_reduce kernel (MLA decode stage-2 reduce/combine) on AMD MI300X / gfx942 so it closes the gap to and outperforms the production HIP kernel aiter.mla_reduce_v1 on GLM-5.2 serving shapes. Use when asked to speed up mla_reduce, close the FlyDSL/HIP mla reduce gap, beat the HIP reduce kernel, add/measure an mla_reduce lever, or run the mla_reduce optimization loop. Combines the autoresearch keep/revert loop with the cdna-kernel-opt lever menu and the KernelForge FlyDSL knowledge base, scoped to this kernel and hardware.
argument-hint: [a lever to try, a scenario to focus (b8_s32 / b1_s128), or nothing to run the loop]
---

# Optimize FlyDSL mla_reduce — beat HIP on GLM-5.2 (gfx942)

An **autonomous, measured optimization loop** for the FlyDSL `mla_reduce` kernel
(`aiter/ops/flydsl/kernels/mla_reduce.py`) on **AMD Instinct MI300X (gfx942 /
CDNA3)**. Stage-2 of split-KV MLA decode: merges per-split partial outputs
`O_i` weighted by `exp(LSE_i − LSE_max)` (online softmax) into the final
bf16/fp16 output. It fuses three methods:

- **autoresearch** — never-stop edit → run → measure → keep-if-better /
  revert-if-not loop, logged to a results table.
- **cdna-kernel-opt** (sibling skill) — diagnose-before-you-change, pull one
  CDNA lever at a time, confirm in the per-iter ISA histogram, never trust a
  single timing number.
- **KernelForge FlyDSL KB** (`~/KernelForge/knowledge_base/flydsl/*`,
  `.../shared/measurement_methodology.md`, `.../shared/cdna4_isa_reference.md`)
  — distilled FlyDSL DSL/pitfalls/LDS knowledge to ground each edit.

**Goal:** the FlyDSL reduce **matches and then beats** the production HIP kernel
`aiter.mla_reduce_v1` on the GLM-5.2 stage-2 decode shapes. Current standing
(graph-mode µs, `19eeb6da1`): **b8_s32 9.0 vs HIP 5.9 (1.53×)**, **b1_s128 25.4
vs 14.2 (1.79×)**, SIMPLE-tier shapes at parity (~1.03×). The gap is the
**massive-tier inner accumulate loop** and **scales with n_splits** — it is
*structural* (how the loop body is realized), not a launch knob.

> The HIP reference is in this tree: `csrc/kernels/mla/reduce.cu`
> (`reduce_output_massive`). **Read it before guessing** — the gap is the gap to
> *that specific loop structure* (see § Why HIP wins). This is the cdna-kernel-opt
> rule: read the reference's source; don't infer it from PMC alone.

---

## The metric and the rules

- **Primary metric:** **CUDA-graph replay latency in µs** (the serving path),
  per scenario, from `~/glm52-mla-reduce/tools/bench_glm52_mla.py`. Lower is better. The headline is the
  **FlyDSL/HIP ratio** per scenario; "beat HIP" means ratio < 1.0. Also record
  kernel-mode (eager) µs as the secondary reference.
- **Correctness gate FIRST, every time:** `op_tests/test_flydsl_mla_reduce.py`
  full tier matrix **plus** the discriminating differential cases (gapped gather
  map, garbage-tail tiles, irregular per-tile splits) that exercise the serving
  OOB guards and the `Tier.ALL` runtime dispatch. AND a **CUDA-graph
  capture/replay** check (no host `.item()` sync — capture-illegal). Speed is
  meaningless until correctness passes.
- **Headline scenarios, both measured:** `b8_s32` (b=8, 32 splits — Jin Tao
  serving steady state, the M64 path) and `b1_s128` (b=1, 128 splits — the M256
  path). Spot-check the SIMPLE tail (`b8_s3`/`b8_s2`) for regressions — it is at
  HIP parity and must stay there.
- **One lever at a time**, on the kernel, gated for correctness, measured in
  graph mode on both headline scenarios, then kept or reverted. **Never stack two
  unverified levers** — fan out in isolated worktrees, then fold (see § Fan-out).
- **Confirm structural levers in the ISA histogram + `vmcnt` depth**, not just
  the stopwatch. On this kernel `s_waitcnt vmcnt(N)` depth in the accumulate loop
  is the single most diagnostic number (see § ISA/ATT check). A change that
  doesn't move the per-iter mnemonic mix or `vmcnt` depth almost certainly did
  nothing.

---

## Environment (read before running anything)

Everything runs **inside the `anguyenh-mla-reduce` docker container** (ROCm 7.2.4,
`rocprofv3` 1.1.0). The optimization worktree is **prod-fold**:

```bash
# the kernel + wrapper + HIP reference live here (branch flydsl-mla-reduce-decode)
WT=/home/anguyenh/aiter-worktrees/prod-fold

# always GPU 5; per-backend JIT dir (NEVER share between hip / flydsl runs)
export HIP_VISIBLE_DEVICES=5
export PYTHONPATH=$WT
export GPU_ARCHS=gfx942        # set so aiter import does not spawn rocminfo
export CU_NUM=304              # avoids rocminfo SIGABRT under rocprofv3
export AITER_JIT_DIR=/tmp/aiter_jit_glm52_optionb   # distinct per backend

# canonical graph+kernel bench, per backend:
docker exec -e HIP_VISIBLE_DEVICES=5 -e PYTHONPATH=$WT anguyenh-mla-reduce \
  python3 /home/anguyenh/glm52-mla-reduce/tools/bench_glm52_mla.py hip       # HIP reference
docker exec ... anguyenh-mla-reduce \
  python3 /home/anguyenh/glm52-mla-reduce/tools/bench_glm52_mla.py optionb   # Tier.ALL production (the one to beat HIP with)
```

`bench_glm52_mla.py` backends: `hip` (reference), `optionb` (`Tier.ALL`,
device-side runtime tier — the production path), `opt5` (host `select_tier`,
pre-Option-B), `baked-mlds` (capture-baking simulation). Optimize **optionb**.

**Confirm the arch is gfx942 before trusting any number:**
`docker exec anguyenh-mla-reduce bash -c 'rocminfo 2>/dev/null | grep -m1 gfx'`.

---

## The op + the kernel surface

Work item = `(head, q-pos-group, reduce-tile)`. A 128-thread block (2 wave64)
owns one `(seq, head)` output row; thread `t` owns `VEC = Dv/128` contiguous
fp32. Fixed GLM-5.2 shape: `H=16, Dv=512` (so **VEC=4**), bf16 out, `max_seqlen_q=1`,
sparse `num_reduce_tile=16384` with 1–8 active tiles, 606-row partial pool.

`compile_mla_reduce` emits one of four per-work-item bodies and (for `Tier.ALL`)
branches on the **device** `n_splits` per tile:

| Tier | n_splits | Body | LSE storage |
|---|---|---|---|
| SIMPLE | < 4 | `emit_simple_body` (register online-softmax) | 0 (regs) |
| M64 | ≤ 64 | `emit_massive_body(nlse=1)` | 1/lane warp0 |
| M256 | ≤ 256 | `emit_massive_body(nlse=4)` | 4/lane warp0 |
| MLDS | > 256 | `emit_massive_body(nlse=5)` | 4 regs + LDS overflow |

`b8_s32` exercises **M64**, `b1_s128` exercises **M256** — both run
`emit_massive_body`. **That function is the optimization target.** SIMPLE is at
parity and untouched.

| File | Role |
|---|---|
| `aiter/ops/flydsl/kernels/mla_reduce.py` | **The kernel** (`emit_massive_body`, `gather_row`, `process_work_item`). Where every lever lands. |
| `aiter/ops/flydsl/mla_reduce_kernels.py` | Python wrapper/dispatcher (capture-safe, no `.item()`). |
| `csrc/kernels/mla/reduce.cu` | **HIP reference** (`reduce_output_massive` = the loop to beat). |
| `op_tests/test_flydsl_mla_reduce.py` | Correctness matrix + differential cases. |
| `op_tests/flydsl_mla_reduce_common.py` | Shared harness / differential reference. |
| `~/glm52-mla-reduce/tools/bench_glm52_mla.py` | GLM-5.2 graph+kernel bench (hip/optionb/opt5/baked-mlds). |

---

## Why HIP wins (the structural gap — `reduce_output_massive`)

Read `csrc/kernels/mla/reduce.cu:306-413`. HIP's massive output accumulate is a
**depth-2 (double-rate) software pipeline** with three properties the FlyDSL
loop currently lacks. These ARE the gap:

1. **Two output `buffer_load`s in flight (depth-2), not one.** HIP processes
   **2 splits per iteration** (`tile_idx += 2`), keeping `oaccu_0` and `oaccu_1`
   outstanding while it FMAs the *previous* pair — `vmcnt(2)`-class overlap.
   FlyDSL's `emit_massive_body` does **depth-1** prefetch (`vmcnt(1)`): one load
   in flight. **This is the #1 lever.**
2. **The gather index is prefetched from LDS ahead of use.** HIP reads
   `p_lds_reduce_partial_map[tile+2]` and `[tile+3]` into registers *before* the
   data load that needs them, so the `buffer_load` address is register-ready —
   no LDS read in the load-address critical path. FlyDSL's `gather_row` reads
   `lds_pmap[split]` at point of use, serializing LDS→address→load.
3. **Uniform buffer descriptor (SGPR rsrc, no waterfall).** HIP builds the gmem
   descriptor once from the kernel-arg pointer (`opus::make_gmem`, SGPRs) so each
   `buf_load_vec` is `buffer_load` with a *uniform* resource + per-lane byte
   offset. Verify FlyDSL's `GTensor` load lowers the same way and is **not**
   emitting a per-lane 64-bit address waterfall (`v_readfirstlane` loop) around
   the gather.
4. **`__builtin_amdgcn_sched_barrier(0)`** pins the schedule between the LSE
   reduce and the output accumulate so the compiler can't move the prefetch
   loads. FlyDSL relies on `hot_loop_scheduler` + `range(init=...)`; the schedule
   must be confirmed in ISA, not assumed.

---

## The loop (autoresearch, scoped to mla_reduce)

Work on a dedicated branch off `flydsl-mla-reduce-decode`. **LOOP until HIP is
beaten on both headline scenarios (or interrupted):**

1. **Baseline FIRST.** Run `bench_glm52_mla.py hip` and `optionb` (graph+kernel),
   record per-scenario FlyDSL µs / HIP µs / ratio / cos into the results table.
   Then profile (§ ISA/ATT check) to classify the bottleneck on *this* HW — for
   the massive loop it is **VMEM-wait-bound** (ATT: ~65% VMEM-wait, ~77% total
   stall), so memory-latency-hiding levers (#1–#4) are the ones that can help.
2. **Pick ONE lever** from the menu, diagnosis-driven (read the ISA / ATT first).
3. **Implement it on `emit_massive_body`** (or `gather_row`). Keep it minimal and
   isolated. Use FlyDSL loop-carried `range(..., init=state)` to add pipeline
   stages; carry the extra in-flight loads + their OOB masks in the loop state
   (the deferred-guard pattern — see § FlyDSL mechanics).
4. **Correctness gate:** `AITER_MLA_REDUCE_FLYDSL=1 python op_tests/test_flydsl_mla_reduce.py`
   (full matrix + differential) AND the CUDA-graph capture/replay check. Any
   failure → fix or revert. Don't measure speed yet.
5. **Measure graph mode** on `b8_s32` and `b1_s128` (same-session A/B vs the
   pre-lever kernel for a clean delta), spot-check `b8_s3` for SIMPLE regression.
6. **Confirm structural levers in ISA** (`vmcnt` depth, `buffer_load` count, no
   new waterfall) — § ISA/ATT check. If the histogram/`vmcnt` didn't move, the
   change is inert; revert.
7. **Keep or revert:**
   - Lower µs with correctness passing → **commit** (`mla_reduce: <lever> (−X% on
     <scenario> graph, ratio A→B)`), update the results table + `glm52-mla-reduce-benchmark-results.md`.
   - Equal/worse, or wins one scenario but regresses another / SIMPLE → **revert**
     (`git checkout -- <file>`), log it as tried-and-discarded with the reason.
8. **Record both outcomes** (wins AND informative failures) so the loop never
   re-tries a dead lever. Go to 2.

**Never stop to ask "should I keep going?"** Pull the next lever. Out of ideas:
re-profile to re-classify, combine two near-misses, or port another structural
detail of `reduce_output_massive`.

---

## The lever menu (each a gfx942 hypothesis; ordered by expected leverage)

For each: implement on `emit_massive_body`, gate correctness, measure both
headline scenarios in graph mode, confirm structural ones in ISA, decide.

| # | Lever | What it does | The question |
|---|---|---|---|
| 1 | **Depth-2 double-rate pipeline** | Process 2 splits/iter, 2 output `buffer_load`s in flight (match HIP `reduce_output_massive`). | Does `vmcnt(2)` overlap cut VMEM-wait below the depth-1 `vmcnt(1)` version? **Highest leverage — start here.** |
| 2 | **Prefetch pmap index from LDS ahead** | Read `lds_pmap[split+depth]` into a reg before the data load that uses it, off the load-address critical path. | Removes the LDS→addr→load serialization HIP avoids. |
| 3 | **Uniform buffer descriptor / kill waterfall** | Ensure the `g_po` gather lowers to `buffer_load` with a uniform SGPR rsrc + per-lane voffset, not a 64-bit per-lane address waterfall. | Check ISA for `v_readfirstlane`/waterfall around the gather; hoist the descriptor if present. |
| 4 | **`sched_barrier` after LSE reduce** | Pin the schedule between `emit_massive_body`'s warp0 LSE reduce + barrier and the output accumulate so prefetch loads aren't sunk. | Does locking the schedule preserve the `vmcnt(N)` overlap the scheduler otherwise undoes? |
| 5 | **Deeper pipeline (depth 3/4)** | Push outstanding loads to `vmcnt_max=4` (ISA shows headroom). | Does going past depth-2 keep helping or blow VGPR / drop occupancy? |
| 6 | **LDS scale round-trip on massive tiers** | The `lds_scale` write→barrier→read in `emit_massive_body` exposes write-read latency (KB Type B). Increase write→read distance or fold. | Insert independent work (the first output prefetch) between the scale write and the barrier. |
| 7 | **waves_per_eu** | Occupancy hint (currently `_DEFAULT_WAVES_PER_EU=4`, opt4). | Re-sweep 1–8 after the pipeline changes register pressure (`AITER_MLA_REDUCE_WAVES_PER_EU`). |
| 8 | **Wider VEC store / epilogue** | Confirm the bf16 epilogue is one `buffer_store_dwordx2/4`, not scalarized `buffer_store_short`. | Already vectorized (opt2) — verify it survived the pipeline edits. |

**Already shipped (do not re-derive):** opt1 persistent grid-stride launch (~14×),
opt2 vectorized epilogue, opt3 tuple loop-carried state, opt4 `waves_per_eu=4`,
opt5 FlyDSL `range`/`hot_loop_scheduler`, **Lever A LDS-stage `reduce_partial_map`**
(−23% b8_s32 graph, shipped `19eeb6da1`), and the **deferred-guard depth-1
prefetch** (in tree; overlap not robustly proven in graph capture).

**Tried and discarded (do not re-try as standalone):** scalarize indptr/pmap/fmap
(Lever B — vgpr −2 but b1_s128 regressed +4%); single-barrier/iteration (Lever C —
b8_s32 +3% regression). See `~/glm52-mla-reduce/shared/mla-reduce-lever-fanout-results.md`.

---

## FlyDSL mechanics for the pipeline levers

The pipeline depth is expressed with a loop-carried `range`. Use the
**deferred-guard pattern** already in `emit_massive_body` (it is why depth-1
reaches `vmcnt(1)`): carry the *raw* loaded `os` vector + a **float OOB mask**
(1.0 in-bounds / 0.0 OOB) in the loop state, issue the next load **before**
consuming the previous, and fold the guard into the scale at the FMA
(`regs[i] + os[i] * (sc * mask)`). Applying the bounds `select` to the loaded
value inside the loop forces `vmcnt(0)` (full drain, zero overlap) — that was the
original broken prefetch.

To go **depth-2**: carry `os0,os1` (2×VEC) + `mask0,mask1` + `regs(VEC)` as state,
prefetch split `s+2` while FMAing `s`, and keep the gather indices for both lanes
register-ready (lever #2). Mirror `reduce_output_massive`'s `oaccu_0/oaccu_1`
double-buffer exactly. Grounding: KernelForge `flydsl/kernel_patterns.md`,
`prefetch-data-load` skill, and the `range(..., init=...)` semantics in
`flydsl/dsl_api.md` (init=None → `scf_range` without iter_args; init=state →
loop-carried).

---

## ISA / ATT check (the source of truth, not the stopwatch)

ATT per-wave aggregates are **unreliable here** — it is a 16384-tile persistent
grid with only ~8 active tiles, so a single-CU trace samples a variable number of
active tiles per wave. **Use deterministic ISA `vmcnt` + CUDA-graph wall-clock as
the verdict;** normalize ATT per os-load execution if used at all.

**ISA (deterministic, primary):** dump `21_final_isa.s` with `FLYDSL_DUMP_IR=1`
and diff the accumulate loop. Reference scripts: `~/glm52-mla-reduce/tools/compare_mla_prefetch_isa.sh`
(prefetch vs no-prefetch metrics). Look for, in the massive accumulate loop:
- `s_waitcnt vmcnt(N)` **depth** — depth-1 prefetch shows `vmcnt(1)`; a depth-2
  win must show `vmcnt(2)` on the `os` vector load. `vmcnt(0)` in the loop body =
  no overlap (the change failed).
- `buffer_load_dwordx4` count — extra prefetched loads should appear; `vmcnt_max`
  (compiler allows up to 4 outstanding — headroom for depth ≤4).
- **No `v_readfirstlane`/waterfall** around the gather (lever #3 regression
  sentinel).
- bf16 epilogue is `buffer_store_dwordx2/4`, not scalarized `buffer_store_short`.

**ATT (when needed, scoped):** kernel name is **`mla_reduce_kernel`** (the
dispatched symbol is `mla_reduce_kernel_0` — a wrong `kernel_include_regex` selects
no dispatch and silently produces no `ui_output_*`). Harness:
`~/glm52-mla-reduce/tools/bench_glm52_trace_b8s32.py`, config `~/glm52-mla-reduce/tools/mla_trace_input.yaml`. Needs
`librocprof-trace-decoder.so` in `/opt/rocm/lib/`. If `ring_buffer: mmap failed`,
lower `att_buffer_size` (container has `ulimit -l = 64 KB`, no huge pages).

```bash
export FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1   # source-map the ISA
python3 ~/glm52-mla-reduce/tools/bench_glm52_trace_b8s32.py      # warmup JIT caches BEFORE rocprof
rocprofv3 -i ~/glm52-mla-reduce/tools/mla_trace_input.yaml -- python3 ~/glm52-mla-reduce/tools/bench_glm52_trace_b8s32.py
python3 ~/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py \
  /tmp/kernel_trace_prefetch/ui_output_agent_*_dispatch_* --topk 10 --mode src --detail
```

---

## Fan-out: parallelize levers, then fold winners

Lever attempts are independent experiments — run up to **4 at once**, each in its
**own git worktree** (mutual exclusivity is mandatory: two subagents must never
edit `mla_reduce.py` in the same tree). The shared GPU is fine (graph bench
serializes on-device). Each subagent runs a full mini-loop on its one lever:
implement → correctness gate (matrix + differential + graph capture) → graph A/B
on both scenarios → confirm `vmcnt`/ISA → return a **structured verdict**
`{lever, files, per-scenario graph µs + ratio + cos, vmcnt-delta, keep/discard,
reason}`. Then **fold**: keep only cos-passing + faster attempts, partition by
compatibility (two levers that both grow VGPR may drop occupancy — don't stack
blindly), apply the compatible winners on a fresh tree and **RE-MEASURE** (the
stacked speedup is not the product of the parts), commit the folded result, fan
out the next batch against the new baseline. **Stop** when a config beats HIP on
both headline scenarios, or a full batch returns no composable win (then
re-profile before the next batch).

---

## Results table (keep updated; do not commit to git)

```
commit	scenario	mode	flydsl_us	hip_us	ratio	cos	status	lever
```

`ratio` = flydsl_us / hip_us (**< 1.0 = beat HIP; that is the target**).
`mode` ∈ {graph, kernel}. `status` ∈ {keep, discard, crash}. One row per
scenario×mode per lever. Fold final standings into
`~/glm52-mla-reduce/glm52-mla-reduce-benchmark-results.md` (the living scoreboard) on each keep.

---

## What does NOT help here (don't waste iterations)

- **Applying the bounds `select` to the loaded value inside the loop** — forces
  `vmcnt(0)`, kills all overlap. Always use the deferred-guard mask fold.
- **Scalarizing indptr/pmap/fmap** (Lever B) — vgpr win but b1_s128 regressed.
- **Moving/removing the massive-body barrier** (Lever C) — no win; b8_s32 regressed.
- **Touching the SIMPLE tier** — it is at HIP parity; changes there only risk
  regressing it. Optimize `emit_massive_body` only.
- **Trusting per-wave ATT aggregates** — workload-variable on this sparse grid;
  use ISA `vmcnt` + graph wall-clock.
- **`baked-mlds` / `opt5` numbers as the target** — they are diagnostics
  (capture-baking, host-tier); the production path to beat HIP with is **optionb**.
- **Compiler-flag sweeps** — structure (pipeline depth, descriptor, schedule) is
  the lever; flags move sub-1% and `s_sched_barrier` can be silently dropped.

---

## Related skills & references

- **cdna-kernel-opt** — the general CDNA lever method (diagnose, pipeline overlap,
  ISA histogram). This skill is its mla_reduce-specialized instance.
- **prefetch-data-load** — loop-carried prefetch / software pipelining patterns
  (levers #1, #5).
- **lds-optimization** — `lds_scale` / `lds_pmap` write-read distance (lever #6).
- **kernel-trace-analysis** / **capture-kernel-trace** — `hotspot_analyzer.py`,
  rocprofv3 ATT capture in the container.
- **debug-flydsl-kernel** — if a lever produces NaN/zeros/wrong output.
- **KernelForge KB:** `~/KernelForge/knowledge_base/flydsl/{dsl_api,kernel_patterns,pitfalls,lds_optimization}.md`,
  `.../shared/{measurement_methodology,cdna4_isa_reference}.md`.
- **Living docs:** `~/glm52-mla-reduce/glm52-mla-reduce-benchmark-results.md` (scoreboard),
  `~/glm52-mla-reduce/shared/mla-reduce-tier-fix-plan.md` (Tier.ALL + follow-up levers),
  `~/glm52-mla-reduce/shared/mla-reduce-lever-fanout-results.md` (A/B/C fan-out),
  `~/glm52-mla-reduce/skills/optimize-mla-reduce-flydsl/mla-reduce-prefetch-trace-results.md` (prefetch trace + deferred-guard fix).

## One-sentence takeaway

> Run the autoresearch keep/revert loop over the cdna-kernel-opt lever menu,
> scoped to `emit_massive_body` on gfx942: baseline first, one lever at a time,
> correctness (matrix + differential + graph capture) before speed, both headline
> scenarios measured in graph mode, structural wins proven by `vmcnt` depth in the
> ISA — because the FlyDSL/HIP gap is the depth-2 double-rate software pipeline
> (with ahead-of-use LDS index prefetch and a uniform SGPR buffer descriptor) that
> HIP's `reduce_output_massive` has and the FlyDSL massive loop does not yet.
