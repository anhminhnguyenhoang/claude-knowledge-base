---
name: optimize-mla-reduce-traversal-flydsl
description: >-
  Autoresearch loop to beat HIP on the FlyDSL mla_reduce GRID-TRAVERSAL FLOOR
  (gfx942 / MI300X) by co-designing the launch config, the persistent work-idx
  decomposition, and the uniform/scalar metadata loads across ALL tiers
  (SIMPLE, M64, M256, MLDS) -- the frontier BEYOND emit_massive_body that the
  accumulate-loop skill optimize-mla-reduce-flydsl leaves on the table. Use when
  the accumulate-loop levers have converged (b1_s128 beats HIP, b8_s32 at
  parity) and the remaining gap is the traversal/prologue lgkmcnt(0) SMEM-wait
  floor -- the mid-split M64 tail (b8_s26/s13/s6/s5 still ~1.13-1.16x HIP) and
  b8_s32 parity. Use when asked to attack the grid-stride traversal, launch grid
  size / occupancy, work-item decomposition, scalar indptr sentinel loads,
  per-work-item barriers, or the kernel prologue of mla_reduce.
disable-model-invocation: true
---

# Optimize FlyDSL mla_reduce TRAVERSAL — beat HIP's floor on GLM-5.2 (gfx942)

An **autonomous, measured optimization loop** for the parts of the FlyDSL
`mla_reduce` kernel (`aiter/ops/flydsl/kernels/mla_reduce.py`) that the sibling
skill **`optimize-mla-reduce-flydsl`** deliberately does NOT touch: the
**launch configuration**, the **persistent grid-stride traversal**, the
**work-item decomposition**, and the **kernel prologue / metadata loads** —
across **all four tiers** (SIMPLE, M64, M256, MLDS), including the
**parity-locked SIMPLE tier**.

The accumulate-loop skill converged: b1_s128 **0.87× (beats HIP)**, b8_s32
**1.00× (parity)**. The residual gap is **structural but outside
`emit_massive_body`** — it is the **grid-traversal + prologue floor**, dominated
by `s_waitcnt lgkmcnt(0)` (LDS/SMEM-wait) on `g_indptr[tile]` sentinel loads and
the kernel-arg / descriptor prologue. This skill attacks *that* floor.

It fuses the same three methods as its parent (autoresearch keep/revert loop;
cdna-kernel-opt diagnose-one-lever-at-a-time; KernelForge FlyDSL KB), but with a
**wider blast radius**: launch geometry and traversal changes affect every tier,
so **SIMPLE parity is now a first-class gated metric, not a spot-check.**

> Read the HIP reference `csrc/kernels/mla/reduce.cu:670-771`
> (`kn_mla_reduce_v1_ps`) before guessing. HIP uses the *same* flat work-idx
> decomposition (`:682-685`) but forces the indptr sentinel loads to **scalar
> SGPR** via `__builtin_amdgcn_readfirstlane` (`:688`, `:690`) and launches
> `num_cu * kOccupancy * 2` blocks (`:1037`). The floor gap is the gap to
> *those* choices — read them; don't infer from PMC alone.

---

## What is different from the parent skill

| | parent (`optimize-mla-reduce-flydsl`) | this skill |
|---|---|---|
| **Target code** | `emit_massive_body`, `gather_row` | `launch_mla_reduce` grid, the `WhileOp` traversal, `process_work_item` metadata/barriers, kernel prologue |
| **Bottleneck** | accumulate loop, VMEM-wait | traversal + prologue, **LDS/SMEM-wait (`lgkmcnt`)** |
| **Tiers affected** | M64/M256 only; SIMPLE untouched | **ALL tiers incl. SIMPLE** |
| **Headline target** | b8_s32 & b1_s128 | **b8_s32 parity→beat** + the **mid-split M64 tail** (b8_s26/s13/s6/s5, still 1.13–1.16× HIP) + hold b1_s128 0.87× |
| **SIMPLE tier** | must not regress (spot-check) | **gated metric — must not regress AND is a beat target** (b8_s3/s2) |
| **ISA verdict number** | `vmcnt(N)` depth in accumulate loop | **`lgkmcnt(0)` count in the traversal**, scalar-vs-vector indptr load, barrier count per active item |

**Do not re-run the accumulate-loop levers here** — pipeline depth, tier-GRP=16,
vectorized LDS reads, and the hoisted-prefetch (#6) are shipped and converged.
This skill only pulls launch/traversal/prologue levers.

---

## Research precedence — KB + forgekernel FIRST, internet LAST

Before searching the web for any technique, DSL API, ISA detail, or launch/
occupancy heuristic, **consult the local KernelForge knowledge base and the
`forgekernel` tool in the devcontainer** (they are curated for exactly this
hardware and DSL, and are offline/authoritative). Only fall back to `WebSearch`
if both come up empty, and say so explicitly when you do.

**Order of consultation for every lever and every open question:**

1. **KernelForge KB (read the markdown directly — primary, always available).**
   `~/KernelForge/knowledge_base/` (host-mounted into the container at the same
   path). Most relevant to this skill:
   - `flydsl/kernel_patterns.md`, `flydsl/dsl_api.md`, `flydsl/dsl_patterns.md`,
     `flydsl/pitfalls.md`, `flydsl/env_vars.md`, `flydsl/compilation_pipeline.md`,
     `flydsl/lds_optimization.md`, `flydsl/kernels_inventory.md`
   - `shared/measurement_methodology.md`, `shared/cdna4_isa_reference.md`,
     `shared/gpu_arch_gfx950.md`, `shared/tuning_rules.md`,
     `shared/cdna4-isa/` (vendored ISA chapters)
   - curated one-shot skills in `knowledge_base/skills/*.json` — for this skill
     especially `flat_grid_block_to_expert_lookup.json` (grid→work mapping),
     `reduce_vgpr_reload_lds.json`, `pmc_before_kernel_rewrite.json`, and the
     `mla_*` entries; browse `skills/skill_index.json` first.

2. **`forgekernel` tool (KernelForge / `kernel-agents` CLI) in the devcontainer.**
   Runs inside `anguyenh-mla-reduce` from `~/KernelForge` (installed as
   `kernel-agents`, or `python3 -m kernel_agents.cli`). Use it to query the KB,
   list what a backend fellow knows, or discover ingestible sources:

   ```bash
   docker exec anguyenh-mla-reduce bash -lc \
     'cd /home/anguyenh/KernelForge && kernel-agents knowledge --backend flydsl'
   docker exec anguyenh-mla-reduce bash -lc \
     'cd /home/anguyenh/KernelForge && kernel-agents fellows && kernel-agents learn sources'
   ```
   For a heavier autonomous sub-campaign on a lever, the `kernel-agents run`
   flydsl fellow is available (bills against Claude Code Max; see
   `~/KernelForge/docs/quickstart.md`). Prefer the KB read + this skill's own
   loop for single levers; reserve `run` for a genuinely open sub-problem.

3. **Internet (`WebSearch`) — last resort only**, when steps 1–2 have no answer.
   Note in the results log that the KB/forgekernel lacked it (a signal to ingest
   it back into the KB later).

---

## The metric and the rules

- **Primary metric:** **CUDA-graph replay latency in µs** per scenario from
  `~/glm52-mla-reduce/tools/bench_glm52_mla.py`; the headline is the FlyDSL/HIP ratio (beat = < 1.0).
  Because these levers are global, **weight the whole scenario sweep**, not two
  headlines: b8_s32, the **mid-split M64 tail (b8_s26, b8_s13, b8_s6, b8_s5 — the
  clearest remaining beat targets)**, b1_s128 (must hold ≤ 0.90×), and the
  **SIMPLE shapes b8_s3 / b8_s2 (must not regress)**.
- **Correctness gate FIRST, every time** — and it matters *more* here: a launch
  or decomposition change can corrupt any tile / any tier or deadlock on a
  divergent barrier. Run `op_tests/test_flydsl_mla_reduce.py` full matrix +
  differential (gapped gather map, garbage-tail tiles, irregular per-tile
  splits) AND the CUDA-graph capture/replay check. Any failure → fix or revert
  before measuring.
- **One lever at a time**, correctness-gated, measured in graph mode across the
  full scenario sweep, then kept or reverted. Never stack two unverified levers.
- **Confirm structural levers in the ISA**, not the stopwatch: the diagnostic
  numbers are the **`lgkmcnt(0)` count**, whether `g_indptr` loads are **scalar
  (`s_load_dword`) vs vector (`buffer_load`/`global_load`)**, the **barrier
  (`s_barrier`) count per active work item**, and the **grid size actually
  launched**. A change that doesn't move these is inert; revert.
- **Beware measurement drift:** the SIMPLE tail drifts ~0.4µs run-to-run.
  Measure every candidate **twice same-session** and treat < ~0.3µs as noise.

---

## Environment (identical to the parent skill)

Everything runs inside the **`anguyenh-mla-reduce`** docker container (ROCm
7.2.4, `rocprofv3` 1.1.0). Worktree **prod-fold**:

```bash
WT=/home/anguyenh/aiter-worktrees/prod-fold
export HIP_VISIBLE_DEVICES=5          # always GPU 5
export PYTHONPATH=$WT
export GPU_ARCHS=gfx942               # avoid rocminfo spawn on import
export CU_NUM=304                     # avoid rocminfo SIGABRT under rocprofv3
export AITER_JIT_DIR=/tmp/aiter_jit_glm52_optionb   # distinct per backend

# canonical graph+kernel bench (grep all scenarios, not just headlines):
docker exec -e HIP_VISIBLE_DEVICES=5 -e PYTHONPATH=$WT -e GPU_ARCHS=gfx942 \
  -e CU_NUM=304 -e AITER_JIT_DIR=/tmp/aiter_jit_glm52_optionb \
  anguyenh-mla-reduce python3 /home/anguyenh/glm52-mla-reduce/tools/bench_glm52_mla.py optionb   # the path to beat HIP with
docker exec ... anguyenh-mla-reduce python3 /home/anguyenh/glm52-mla-reduce/tools/bench_glm52_mla.py hip
```

Optimize **optionb** (`Tier.ALL`, device-side runtime tier — production path).
Confirm arch: `docker exec anguyenh-mla-reduce bash -c 'rocminfo 2>/dev/null | grep -m1 gfx'`.

---

## The traversal + launch surface (where every lever lands)

```
launch_mla_reduce           (mla_reduce.py:858-922)  -- GRID/BLOCK config
  persistent grid = max_splits * OCC*2   (:893-898)  -- OCC=8, block=128 (2 wave64)
  mla_reduce_kernel(...)                              -- the emitted kernel
    prologue                (mla_reduce.py:280-310)   -- kernel-arg s_load, GTensor
                                                          descriptors, LDS STensors
    persistent WhileOp      (mla_reduce.py:813-849)   -- work_idx = block_idx,
                                                          stride grid_dim.x
      decomposition         (:824-827)  head=work%H (fastest), block_idx(NTG),
                                          tile = ... (slowest)
      sentinel load+test    (:829-835)  tile_start = g_indptr[tile]; == last ?
      if not_past_end:      (:836-844)  process_work_item(...) + barrier(:843)
      next_work             (:846-848)  is_past_end ? tot_work : work_idx+stride
    process_work_item       (mla_reduce.py:312-...)   -- runs for ALL tiers
      t0,t1 = g_indptr[tile], g_indptr[tile+1]  (:319-320)
      stage reduce_partial_map -> LDS + barrier  (:325-332)   <-- barrier #1
      has_work / q-range guards                  (:339-352)
      tier dispatch: emit_simple_body | emit_massive_body(nlse=1|4|5)
```

Constants: `NUM_THREADS=128`, `WARP=64`, `OCC=8`, `LDS_MAX_SPLITS=304`.
HIP counterpart: `kn_mla_reduce_v1_ps` (`reduce.cu:670-771`); HIP launch
`ps_grid_size = num_cu * kOccupancy * 2` (`reduce.cu:1037`).

| File | Role |
|---|---|
| `aiter/ops/flydsl/kernels/mla_reduce.py` | kernel: `launch_mla_reduce`, traversal `WhileOp`, `process_work_item`, prologue. **Every lever lands here.** |
| `aiter/ops/flydsl/mla_reduce_kernels.py` | wrapper/dispatcher (`num_cu`, `ps_grid_size`, capture-safe). |
| `csrc/kernels/mla/reduce.cu` | HIP reference `kn_mla_reduce_v1_ps` + `reduce_output_massive`. |
| `op_tests/test_flydsl_mla_reduce.py` | correctness matrix + differential. |
| `~/glm52-mla-reduce/tools/bench_glm52_mla.py` | GLM-5.2 graph+kernel bench. |
| `~/glm52-mla-reduce/tools/isa_metrics_mla.sh` | dump `21_final_isa.s` + ISA metric counts (see below). |
| `~/glm52-mla-reduce/tools/bench_glm52_trace_b8s32.py`, `~/glm52-mla-reduce/tools/mla_trace_input.yaml` | ATT capture harness. |

---

## Why the floor exists (the diagnosis to build on)

From the ATT re-profile of b8_s32 (`hotspot_analyzer.py`, src-mapped) — the
dominant stall is **`s_waitcnt lgkmcnt(0)` (LDS/SMEM-wait)**, NOT VMEM:

| Src line | Role | Stall |
|---|---|---|
| `mla_reduce.py:813` | grid-stride `WhileOp` (indptr sentinel) | 15.9K `lgkmcnt(0)`, 86% |
| `mla_reduce.py:809` | traversal setup | 11.1K `lgkmcnt(0)`, 92% |
| `mla_reduce.py:302` | buffer-resource / descriptor build | 22K `lgkmcnt(0)` |
| `mla_reduce.py:280` | kernel-arg `s_load_dwordx4` | 2.8K SMEM |

With only 8 active tiles / 16384, b8_s32's wall-clock is **prologue + traversal
+ 8 tiny reductions**. Cross-check: SIMPLE floor b8_s3 = 4.0µs ≈ pure
traversal+launch, so only ~2µs is the 8 M64 reductions; that ~4µs floor is
shared with HIP (HIP b8_s3 = 3.8µs) — which is *why* b8_s32 sits at parity.
**To beat HIP on b8_s32 and the mid-split tail, the floor must drop.**

---

## The loop (autoresearch, scoped to launch/traversal/prologue)

Work on a dedicated branch off `flydsl-mla-reduce-decode` (or the parent's latest
shipped commit). **LOOP until HIP is beaten on b8_s32 + the mid-split tail while
b1_s128 holds ≤ 0.90× and SIMPLE does not regress (or interrupted):**

1. **Baseline:** reuse the last committed scoreboard if nothing changed
   (`~/glm52-mla-reduce/glm52-mla-reduce-benchmark-results.md`); otherwise bench `optionb` + `hip`
   graph+kernel across the full scenario sweep. Profile only when the code
   changed (ATT re-profile / ISA metrics).
2. **Pick ONE lever** from the menu, diagnosis-driven. **Ground it in the KB /
   forgekernel first** (see § Research precedence) — read the relevant
   `flydsl/*` + `shared/*` playbook (and any matching `knowledge_base/skills/*`
   entry) before writing code; only WebSearch if the KB/forgekernel lack it.
3. **Implement** it minimally on `launch_mla_reduce` / the `WhileOp` / the
   prologue / `process_work_item`. Keep tier semantics identical.
4. **Correctness gate:** full matrix + differential + CUDA-graph capture/replay.
   Fix or revert on any failure. Don't measure speed yet.
5. **Measure graph mode twice** across the full sweep (b8_s32, b8_s26, b8_s13,
   b8_s6, b8_s5, b1_s128, **b8_s3, b8_s2**). A/B vs the pre-lever kernel.
6. **Confirm in ISA** (`~/glm52-mla-reduce/tools/isa_metrics_mla.sh`): `lgkmcnt(0)` count, scalar-vs-
   vector indptr load, `s_barrier` count, grid size. Inert → revert.
7. **Keep or revert:**
   - Lower µs on target shapes, correctness passing, **b1_s128 ≤ 0.90× and
     SIMPLE not regressed** → **commit** (`mla_reduce: <lever> (−X% on <shapes>
     graph, ratio A→B)`), update the scoreboard.
   - Wins one shape but regresses another / SIMPLE / b1_s128 → **revert**, log as
     tried-and-discarded with the reason.
8. **Record both outcomes** so the loop never re-tries a dead lever. Go to 2.

**Never stop to ask "should I keep going?"** Out of ideas: re-profile to
re-classify, combine two near-misses, or port another structural detail of
`kn_mla_reduce_v1_ps`.

---

## The lever menu (launch + traversal + prologue; ordered by expected leverage)

| # | Lever | What it does | The question |
|---|---|---|---|
| T1 | **Scalar/uniform indptr sentinel load** | Force `g_indptr[tile]`/`t0`/`t1` to a **scalar SGPR** load (mirror HIP `__builtin_amdgcn_readfirstlane`, `reduce.cu:688/690`) so the sentinel test is `s_load_dword`, not a per-lane vector load with `lgkmcnt`. | Does the traversal `lgkmcnt(0)` count drop? **Highest leverage — start here;** it is the single biggest floor stall and HIP's explicit choice. |
| T2 | **Launch grid-size / occupancy sweep** | Retune the persistent grid (`:893-898`, currently `max_splits*OCC*2`, OCC=8). Too many blocks → most do one sentinel-load then terminate (wasted launch); too few → active work under-parallelized. Sweep OCC and the ×2, and confirm the grid actually matches `num_cu*OCC*2`. | Is the grid mis-sized for the sparse 8-active-tile profile vs HIP's `num_cu*kOccupancy*2`? |
| T3 | **Work-idx decomposition reorder** | The flat decomposition (`:824-827`) puts `head` fastest, `tile` slowest, so a block reloads `g_indptr[tile]` for each (head,ntg) of the same tile. Reorder so `tile` co-locates within a block-stride (or stage the sentinel once per tile) to cut redundant sentinel loads. | Can the sentinel load count per active tile drop below H×NTG without breaking early-termination? |
| T4 | **Stage indptr sentinel to LDS/regs once** | Pre-load the (small) active indptr prefix into LDS or a scalar cache in the prologue, so the traversal tests a cheap LDS/reg value instead of a global load per work_idx. | Does trading the per-iter global sentinel load for a one-time staged read cut the floor? (mirrors Lever-A pmap staging, applied to indptr.) |
| T5 | **Cut a per-active-item barrier** | There are two barriers per active item: pmap-stage (`:332`) + LDS-reuse fence (`:843`). HIP does one `s_barrier` at loop-top (`reduce.cu:761`). Restructure so only one is needed (e.g. single top-of-iteration fence). | Does one barrier/iteration hold correctness (differential + graph) and cut the SIMPLE/mid-split floor? (Parent's Lever C regressed +3% with a *different* placement — try HIP's top-of-loop form, not removal.) |
| T6 | **Prologue descriptor hoist / scalarize** | The kernel-arg `s_load` + GTensor buffer-descriptor build (`:280`, `:300-310`) is a fixed `lgkmcnt` prologue cost. Ensure descriptors are built once from uniform SGPR pointers and not rematerialized in the loop. | Is any descriptor/kernel-arg load sunk into the traversal? Hoist it. |
| T7 | **Early-out ordering** | Order the sentinel/`has_work` tests so inactive tiles exit with the *fewest* memory ops (test `tile_start==last` before any pmap-stage/`t1` load). | Do inactive work_idx iterations touch memory they don't need? |
| T8 | **Grid-stride vs tile-partition launch** | Alternative: give each block a contiguous tile range (block→tile map) instead of flat grid-stride, so indptr access is contiguous/coalesced per block. Larger change; fan out in a worktree. | Does a tile-partitioned launch beat flat grid-stride on the sparse profile? |

**Already shipped / converged (do not re-derive):** everything in
`emit_massive_body` (depth-2/GRP pipeline, tier-GRP=16, vectorized LDS reads,
hoisted os-prefetch #6); persistent grid-stride launch itself; LDS pmap staging.

**Tried and discarded (do not re-try as standalone):** `waves_per_eu` re-sweep
(inert — wpe 1..8 identical); indptr-load dedup passing `tile_start` into
`process_work_item` (inert — compiler already CSE's the two identical loads);
single-barrier via *removal* (parent Lever C, b8_s32 +3%). See the scoreboard.

---

## ISA / ATT check (the source of truth)

**ISA (deterministic, primary):** `~/glm52-mla-reduce/tools/isa_metrics_mla.sh <tag> <splits>` dumps
`21_final_isa.s` and reports `buffer_load`, `vmcnt`, `lgkmcnt`-relevant counts,
`v_readfirstlane`, vgpr. For traversal levers the numbers that matter:

- **`lgkmcnt(0)` count** in/around the traversal — T1/T3/T4 must lower it.
- **indptr load form** — `grep` the ISA for how `g_indptr[tile]` lowers: a
  **scalar `s_load_dword`** (T1 win) vs a vector `global_load`/`buffer_load`
  with a following `s_waitcnt lgkmcnt(0)` (the floor stall).
- **`s_barrier` count** per active work item — T5 must reduce it (and stay
  correct).
- **grid size** — confirm `launch` grid equals the intended `num_cu*OCC*2`.

```bash
docker exec -e HIP_VISIBLE_DEVICES=5 -e GPU_ARCHS=gfx942 -e CU_NUM=304 \
  anguyenh-mla-reduce bash /home/anguyenh/glm52-mla-reduce/tools/isa_metrics_mla.sh cur 32
docker exec anguyenh-mla-reduce bash -c \
  "grep -n -E 's_barrier|s_load_dword|global_load_dword.*indptr|lgkmcnt' /tmp/mla_cur_isa.s | head"
```

**ATT (scoped, when re-classifying):** warm JIT first, then rocprofv3; the trace
dir is **inside the container** at `/tmp/kernel_trace_prefetch/ui_output_agent_*_dispatch_*`
(agent `_48720_` = GPU 5). Per-wave aggregates are unreliable on this sparse
grid — use the **src-mapped stall-type distribution** (is it `lgkmcnt` in the
traversal?) qualitatively, and the ISA + graph wall-clock as the verdict.

```bash
docker exec ... -e FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 anguyenh-mla-reduce \
  python3 /home/anguyenh/glm52-mla-reduce/tools/bench_glm52_trace_b8s32.py                    # warmup JIT
docker exec ... anguyenh-mla-reduce bash -c \
  'cd /home/anguyenh && rocprofv3 -i /home/anguyenh/glm52-mla-reduce/tools/mla_trace_input.yaml -- \
   python3 /home/anguyenh/glm52-mla-reduce/tools/bench_glm52_trace_b8s32.py'
docker exec ... anguyenh-mla-reduce python3 \
  ~/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py \
  /tmp/kernel_trace_prefetch/ui_output_agent_48720_dispatch_32768 --topk 12 --mode src --detail
```

---

## Fan-out: parallelize levers, then fold

Launch/traversal levers are independent experiments — run up to **4 at once**,
each in its **own git worktree** (two subagents must never edit `mla_reduce.py`
in the same tree). Each subagent runs a full mini-loop on its one lever:
implement → correctness gate (matrix + differential + graph capture) → graph A/B
across the **full sweep incl. SIMPLE** → confirm ISA (`lgkmcnt`, indptr form,
barrier count) → return a structured verdict `{lever, files, per-scenario graph
µs + ratio + cos, lgkmcnt/barrier delta, keep/discard, reason}`. Then **fold**:
keep only cos-passing + faster-without-SIMPLE/b1_s128-regression attempts, apply
compatible winners on a fresh tree and **RE-MEASURE** the full sweep, commit,
fan out the next batch. **Stop** when a config beats HIP on b8_s32 + the
mid-split tail (SIMPLE held, b1_s128 ≤ 0.90×), or a full batch returns no
composable win (then re-profile before the next batch).

---

## Results table (keep updated; do not commit to git)

```
commit  scenario  mode  flydsl_us  hip_us  ratio  cos  status  lever
```

`ratio` = flydsl_us / hip_us (< 1.0 = beat HIP). One row per scenario×mode per
lever — **include b8_s3 / b8_s2 every time** (SIMPLE is now a gated metric). Fold
final standings into `~/glm52-mla-reduce/glm52-mla-reduce-benchmark-results.md` on each keep.

---

## What does NOT help here (don't waste iterations)

- **Touching `emit_massive_body`** — that is the parent skill's converged domain;
  this skill's edits are launch/traversal/prologue/decomposition only.
- **`waves_per_eu` sweeps** — confirmed inert (wpe 1..8 identical).
- **Source-level indptr-load dedup** — the compiler already CSE's identical
  `g_indptr[tile]` loads (no store between); force *scalar* (T1) instead.
- **Barrier *removal*** — parent Lever C regressed b8_s32 +3%. Only try HIP's
  top-of-loop single-`s_barrier` restructure (T5), correctness-gated.
- **Regressing SIMPLE or b1_s128 for a mid-split win** — the win must be net
  across the sweep; SIMPLE parity and b1_s128 ≤ 0.90× are hard constraints.
- **Trusting per-wave ATT aggregates** — sparse grid; use the src-mapped stall
  *type*, ISA counts, and graph wall-clock.
- **Compiler-flag sweeps** — structure (grid, decomposition, scalar loads,
  barriers) is the lever; flags move sub-1%.

---

## Related skills & references

- **`optimize-mla-reduce-flydsl`** — the sibling accumulate-loop skill; its
  converged state (b1_s128 0.87×, b8_s32 1.00×) is this skill's baseline.
- **cdna-kernel-opt** — general CDNA diagnose-one-lever method.
- **kernel-trace-analysis** / **capture-kernel-trace** — `hotspot_analyzer.py`,
  rocprofv3 ATT capture in the container.
- **lds-optimization** — for T4 indptr-staging write-read distance.
- **debug-flydsl-kernel** — if a launch/traversal change yields NaN/zeros/wrong
  output or a barrier deadlock.
- **KernelForge KB + `forgekernel` tool (consult FIRST — see § Research
  precedence):** `~/KernelForge/knowledge_base/flydsl/{dsl_api,kernel_patterns,
  dsl_patterns,pitfalls,lds_optimization,env_vars,compilation_pipeline}.md`,
  `.../shared/{measurement_methodology,cdna4_isa_reference,gpu_arch_gfx950,tuning_rules}.md`,
  `.../skills/*.json`; query via `kernel-agents knowledge --backend flydsl` in
  the `anguyenh-mla-reduce` devcontainer. Internet is the last resort.
- **Living docs:** `~/glm52-mla-reduce/glm52-mla-reduce-benchmark-results.md` (scoreboard incl. the
  traversal re-profile + discarded launch levers), `~/glm52-mla-reduce/shared/mla-reduce-tier-fix-plan.md`.

## One-sentence takeaway

> Run the autoresearch keep/revert loop over the launch + traversal + prologue
> lever menu across ALL tiers on gfx942 — baseline first, one lever at a time,
> correctness (matrix + differential + graph capture) before speed, the FULL
> scenario sweep (SIMPLE and b1_s128 as hard constraints) measured in graph mode,
> structural wins proven by the traversal `lgkmcnt(0)` count / scalar indptr load
> / barrier count in the ISA — because the residual FlyDSL/HIP gap is no longer
> the accumulate loop but the grid-traversal + prologue floor that HIP's
> `kn_mla_reduce_v1_ps` beats with scalar (`readfirstlane`) sentinel loads, a
> tuned `num_cu*OCC*2` launch, and a single top-of-loop barrier.
