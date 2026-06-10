---
name: jdbba-autoresearch
description: Autonomous experiment loop to optimize the FlyDSL jagged_dense_bmm_broadcast_add (jdbba) kernel on AMD MI300X / gfx942, targeting up to 2x over the current baseline (the Triton reference / Meta's internal CUTLASS gap). Use when asked to speed up jdbba, run the jdbba optimization loop, re-tune jdbba for gfx942, add/measure a jdbba lever, or "do autoresearch on the jagged bmm kernel". Combines the autoresearch edit→run→measure→keep/discard loop with the cdna-kernel-opt lever menu, scoped to this specific kernel and hardware.
argument-hint: [a lever to try, a shape to focus, or nothing to run the loop]
---

# jdbba Autoresearch — optimize the jagged bmm on MI300X (gfx942)

An **autonomous, measured optimization loop** for the FlyDSL
`jagged_dense_bmm_broadcast_add` (jdbba) kernel on **AMD Instinct MI300X
(gfx942 / CDNA3)**. It fuses two methods:

- **autoresearch** (`~/autoresearch/program.md`): the never-stop
  edit → commit → run → measure → keep-if-better / revert-if-not loop, logged to
  a results table.
- **cdna-kernel-opt** (the sibling skill): diagnose-before-you-change, pull one
  CDNA lever at a time, confirm in the per-iter ISA histogram, never trust a
  single timing number.

**Goal:** close the Triton→CUTLASS gap. Meta's Triton kernel is ~20% slower on
MI300X than H100; Meta's internal CUTLASS is ~2x faster than Triton. We can't see
CUTLASS, so the target is **up to 2x over the current FlyDSL/Triton baseline** on
the four headline shapes, measured on gfx942.

> **The dispatch table is arch-keyed** (`jagged_dense_bmm_dispatch_v2.json`,
> `arch-keyed-v1` schema). The committed **gfx942 section is the official
> baseline** (empty winners → heuristic, `use_mfma_k32: false`); the gfx950
> section holds the older MI355X winners. Filling in per-shape **gfx942** winners
> as levers land is the open work — that is exactly what this loop does.

---

## The metric and the rules

- **Primary metric:** kernel time in **ms**, FlyDSL vs Triton, from
  `triton.testing.do_bench` (CUDA-event, L2-flushed cold-L2 — the
  deployment-representative number). Lower is better. The headline is the
  **FlyDSL/Triton ratio** per shape; "2x" means FlyDSL ≈ 0.5× Triton.
- **Correctness gate FIRST, every time:** `cos ≥ 0.999` vs torch eager on all 4
  shapes AND the edge cases (empty group `M_b=0`, partial bottom tile
  `M_b % BLOCK_M ≠ 0`, one-long-many-short). Speed is meaningless until cos
  passes. The bench's `-test` flag prints `PASS/FAIL` + cos per shape.
- **Two regimes, both measured:** `uniform` (M_i = max_seq_len, zero tail waste,
  the controlled baseline) and `skew` (deployment distribution, ~20-30% empty
  groups). Several levers help one regime and hurt the other — record both.
- **One lever at a time**, on the kernel, gated for correctness, measured in both
  regimes, then kept or reverted. **Never stack two unverified levers.**
- **Confirm structural changes in the ISA histogram**, not just the stopwatch — a
  change that doesn't move the per-iter mnemonic count almost certainly did
  nothing (see cdna-kernel-opt Step 3).

---

## Environment (read this before running anything)

Everything runs **inside the `jdbba-flydsl` docker container**. torch / triton /
flydsl / aiter / generative-recommenders are in the container, NOT on the bare
host. The host `~/aiter` checkout is bind-mounted at **`/workspaces/meta/aiter`**
inside the container (same git checkout — edits on the host are seen in the
container immediately, no copy needed).

```bash
# confirm the arch is gfx942 (NOT gfx950) before trusting any number
docker exec jdbba-flydsl bash -c 'rocminfo 2>/dev/null | grep -m1 gfx'   # -> gfx942

# the canonical bench invocation (uniform, all 4 shapes, with correctness):
docker exec -e PYTHONPATH=/workspaces/meta/aiter \
  -w /workspaces/meta/aiter jdbba-flydsl \
  python3 op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py --metric time -test

# both regimes through the production dispatch:
docker exec -e PYTHONPATH=/workspaces/meta/aiter \
  -w /workspaces/meta/aiter jdbba-flydsl \
  python3 op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py --regime both -test

# one custom shape, FlyDSL only (fast iteration):
docker exec -e PYTHONPATH=/workspaces/meta/aiter \
  -w /workspaces/meta/aiter jdbba-flydsl \
  python3 op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py \
  -b 120 -d 256 -kout 256 --metric time -test --flydsl-only
```

genrec (the Triton baseline) imports as
`generative_recommenders.ops.triton.triton_jagged` — already on the container
PYTHONPATH. If the Triton provider is unavailable the bench warns and runs FlyDSL
only; you lose the ratio, so fix it before trusting a "win".

**Known startup blocker #1 — stale aiter JIT build.** If the bench dies with
`ImportError: cannot import name 'MxScaleRoundMode' from aiter.utility.mx_types`,
the prebuilt `aiter/jit/module_aiter_core.so` is stale vs the header
`csrc/include/rocm_ops.hpp` (which DOES define the binding). A plain
`AITER_REBUILD=1`, or deleting just the `.so`, does NOT fix it (ccache reuses the
cached `aiter_core_pybind.cu` object — the "rebuild" finishes in ~7s and still
fails). You must **wipe the build dir** so the translation unit recompiles
(~24s, verified working):

```bash
docker exec -w /workspaces/meta/aiter jdbba-flydsl bash -c \
  'rm -rf aiter/jit/build/module_aiter_core aiter/jit/module_aiter_core.so && python3 -c "import aiter"'
```

This is a pre-existing aiter env issue, unrelated to jdbba.

**Dispatch JSON is now arch-keyed (resolved blocker — was a gfx942 crash).** The
`jagged_dense_bmm_dispatch_v2.json` uses the `arch-keyed-v1` schema:
`{schema, by_arch: {gfx942: {...}, gfx950: {...}}}`. The loader auto-selects the
section matching the detected arch (`flydsl.runtime.device.get_rocm_arch`), so on
MI300X you get the **gfx942** section automatically — no env var needed.
Historically the flat JSON was gfx950-only and forced `use_mfma_k32: true`, which
**core-dumped** on gfx942
(`LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.mfma.f32.16x16x32.bf16`);
the arch split fixes that. The committed **gfx942 section is the official
baseline**: empty `winners` → D-bucketed heuristic with `use_mfma_k32: false`,
`xcd_c/xcd_w` null (kernel defaults). That is the reproducible pre-tuning
baseline; per-shape gfx942 winners get filled in by the loop as levers land.

```bash
# out-of-box: no env override needed, auto-selects gfx942
docker exec -e PYTHONPATH=/workspaces/meta/aiter \
  -w /workspaces/meta/aiter jdbba-flydsl \
  python3 op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py --regime both --metric time -test
```

Overrides still work: `FLYDSL_JAGGED_DENSE_BMM_DISPATCH_V2_JSON=<file>` (accepts
either the arch-keyed or a flat legacy schema, the file must be readable *inside*
the container), and `FLYDSL_JAGGED_DENSE_BMM_ARCH=<arch>` forces a section.

**Official gfx942 baseline (2026-06-10, `do_bench` cold-L2 ms, flydsl/triton, all
cos=1.0000):**

| shape | uniform flydsl | uniform triton | skew flydsl | skew triton |
|---|---|---|---|---|
| B120_D256  | 0.587  | 0.642  | 0.151 | 0.154 |
| B120_D512  | 1.476  | 1.877  | 0.349 | 0.402 |
| B1024_D256 | 4.820  | 5.380  | 1.046 | 1.056 |
| B1024_D512 | 12.302 | 16.023 | 2.613 | 3.080 |

FlyDSL already leads Triton 1.0-1.3×; the 2× target is the remaining gap. Most
headroom is the **D256** shapes (1.01-1.12×); skew B1024_D256 is ~tied.

---

## The loop (autoresearch, scoped to jdbba)

Work on a dedicated branch off `main`. The jdbba history lives in the
`anguyenh/flydsl-*` line; commit each kept lever.

**LOOP FOREVER (until interrupted):**

1. **Establish the gfx942 baseline FIRST** (the autoresearch "first run is the
   baseline" rule). Run the canonical bench in both regimes, record per-shape
   FlyDSL ms / Triton ms / ratio / cos into a results table (see below). Do NOT
   compare to any gfx950 number. Also compute the **bound analysis** (§ Bound
   check) so you know if you're memory- or compute-bound on *this* HW — it tells
   you which levers can possibly help.
2. **Pick ONE lever** from the menu. Prefer the diagnosis-driven one (run a
   rocprof PMC pass / read the ISA first; don't guess).
3. **Implement it on the kernel** (`jagged_dense_bmm_gen.py` for static, or the
   dispatch / persistent kernel — see § Files & knobs). Keep the change minimal
   and isolated to that one lever.
4. **Correctness gate:** run the bench with `-test`. If any shape/edge-case
   `cos < 0.999`, the lever is wrong — fix or revert. Don't measure speed yet.
5. **Measure both regimes** with `do_bench`. For an L2-reuse-sensitive lever
   (anything touching the XCD remap), also take a `rocprofv3 --kernel-trace` p10
   (hot-L2) reading — cold vs hot L2 can flip the verdict.
6. **Confirm structural levers in the ISA histogram** (§ ISA check). If the
   per-iter mnemonic mix didn't move, the change is inert.
7. **Keep or revert:**
   - Improved (lower ms) with cos passing → **commit** ("[FlyDSL] jdbba: <lever>
     (+X% on <shape/regime>)"), update the results table and, if it's a per-shape
     win, the dispatch JSON winner.
   - Equal/worse, or helps one regime but hurts the other more → **revert**
     (`git checkout -- <file>`), log it as tried-and-discarded with the reason.
8. **Record both outcomes** — wins AND informative failures — so the loop doesn't
   re-try a dead lever. Then go to 2.

**Never stop to ask "should I keep going?"** — pull the next lever. If you run out
of ideas: re-read the per-lever results, combine two near-misses, read the Triton
reference's autotune space (`triton_jagged.py` lines 268-292) for a tile/warp
config you haven't tried, or re-profile to re-classify the bottleneck.

**Automate the sweep, don't hand-run it.** Levers that have a discrete grid
(XCD remap, BLOCK_K/M/N, STAGES, warps, waves_per_eu) are searched with the
**autotuning scripts** below — a script enumerates the grid, gates cos, times
both regimes, and emits the winners. The keep/revert loop above still governs
*which lever* and *whether to promote*; the script is just how you cover that
lever's grid reproducibly. Build the script as part of working the lever, commit
it next to the bench, and re-run it on the next arch.

---

## Fan-out: parallelize the levers, then fold the winners

The loop above is sequential by *attribution* (one lever's effect must be cleanly
measurable), **not** by wall-clock. Lever attempts are independent experiments, so
run them concurrently and only serialize the decision.

**Fan out (up to 4 at once).** Pick up to **4 mutually-exclusive attempts** — each
a *different lever*, or *different points of one lever's grid that need source
edits* (e.g. four BLOCK_M/BLOCK_N tile choices). Spawn one **subagent per attempt**
(use the Agent tool; send all in a single message so they run in parallel). Mutual
exclusivity is mandatory: two subagents must never edit the same kernel source, or
their measurements contaminate each other.

- **Isolation is required.** Give each subagent its **own git worktree/clone**
  (`isolation: "worktree"`) so concurrent kernel edits don't collide. The shared
  GPU is fine — `do_bench` serializes on the device — but the *source trees* must
  be separate.
- **Each subagent does a full mini-loop on its one attempt:** implement the lever
  on its clone → cos-gate (`cos≥0.999`, all 4 shapes + edge cases) → measure both
  regimes with `do_bench` (+ rocprofv3 p10 if L2-sensitive) → confirm structural
  levers in the ISA histogram.
- **Each returns a structured verdict**, not prose: `{lever, files_touched,
  per-shape×regime ms + cos, ratio vs baseline, keep/discard, ISA-delta note}`.
  Discards come back too (with the reason) so the loop never re-tries a dead lever.

**Fold the winners (combine whatever composes).** Fan-out finds *individually*
winning levers; the 2x prize is in **stacking** them (see *Honest ceiling*). After
the batch returns:
1. Keep only the subagent attempts that passed cos AND beat the baseline ms.
2. **Partition by compatibility.** Levers that touch disjoint mechanisms compose
   (e.g. XCD remap #6 + wide-store epilogue #7 + B-stationary #12 + a tile choice
   #3). Levers that fight for the same resource do **not** stack blindly — two that
   both grow LDS can blow the **64 KB** ceiling; two tile-shape changes conflict.
   Use the bound check + `smem_bytes` to rule out illegal combinations.
3. **Apply the compatible winners together on a fresh clone and RE-MEASURE.** The
   combined speedup is *not* the product of the individual ones — interactions
   (LDS pressure, occupancy, register spills) can erase a win. Only the
   re-measured stacked number counts. If a pair regresses when combined, drop the
   lower-leverage one and re-measure.
4. **Commit the folded result** as one promotion (update the dispatch JSON winner +
   results table), then fan out the next batch of 4 against this new baseline.

**Stop condition:** keep fanning out + folding until a stacked config hits the **2x
ratio** on a shape/regime, or a full batch of 4 returns no composable win (then
re-profile to re-classify the bottleneck before the next batch). Use the autotuning
scripts *inside* a subagent when its attempt is a grid (one subagent owns one
lever's whole grid); use raw single-config attempts when it's a one-off source edit.

**Timeout:** a single bench invocation over 4 shapes is well under a minute once
compiled; first compile of a new (shape, knob) tuple can take a couple minutes
(FlyDSL JIT). If a run hangs past ~10 min, kill it and treat the lever as a
failure.

---

## The op (one-paragraph spec)

For each group `b` over its packed row slice `[s,e) = [seq_offsets[b], seq_offsets[b+1])`:
`Out[s:e,:] = Jagged[s:e,:] @ Dense[b] + Bias[b][None,:]`, i.e.
`(M_b×N) = (M_b×K)·(K×N) + (1×N broadcast)`. BF16 in/out, FP32 accumulate, all
row-major. **The one hard fact:** group boundaries (`seq_offsets`, a `B+1` prefix
sum) are **device-resident** — the host doesn't know `M_b` at launch, so the
group→row mapping is resolved on the GPU.

**⚠️ Naming clash.** The HSTU bench uses `(B, D, K, N)`:
- bench **B** = number of groups
- bench **D** = reduction K (the GEMM contraction dim)
- bench **K** (`Kout`) = output N
- bench **N** = max_seq_len (the M-envelope)

So `B1024_D512_K512` means 1024 groups, reduction K=512, output N=512. The kernel
code and `cdna-kernel-opt` use *standard* GEMM meaning. Keep them straight.

### Headline shapes (max_seq_len Mi = 7680, a tile multiple near the deployment mean)

| shape | B groups | D (reduction K) | Kout (output N) | regime |
|---|---|---|---|---|
| B120_D256  | 120  | 256 | 256 | inference, small |
| B120_D512  | 120  | 512 | 512 | inference, large |
| B1024_D256 | 1024 | 256 | 256 | train, small |
| B1024_D512 | 1024 | 512 | 512 | train, large |

### Shape facts that drive every decision

- **Many tiles** (M_i≈7800 × many groups → 14k-125k tiles) → no occupancy
  problem → **split-K is unnecessary, keep k_batch=1.**
- **Tiny reduction (D=256/512), BLOCK_K=64-128 → K-loop is only 4-8 steps** →
  per-block setup + epilogue is a *large* fraction of runtime → the win is
  **amortizing fixed costs**, not flop scheduling.
- **Dense[b] is 128KB-512KB, reused across ~61 M-tiles of its group → L2-resident**,
  not an HBM bottleneck. The kernel is **memory-bound on A/Out streaming.**
- **No multi-weight A-sharing fusion** is available — the model graph
  (`ContextualizedMLP`) computes Dense[b] fresh per step and doesn't multiply the
  jagged input by several weights. Don't plan around §8.5-style fusion.

---

## Files & knobs (the search space)

| File | Role |
|---|---|
| `aiter/ops/flydsl/kernels/jagged_dense_bmm_gen.py` | **Main static kernel factory** (`_build_launcher` memoized per shape/knob). Where most levers land. |
| `aiter/ops/flydsl/kernels/jagged_dense_bmm_persist_dev.py` | Persistent on-device problem-visitor (skew candidate, no host sync). |
| `aiter/ops/flydsl/jagged_dense_bmm_dispatch_v2.py` | Production dispatch: per-shape config + skew gate (persist vs static) + regime XCD gate. |
| `aiter/ops/flydsl/jagged_dense_bmm_dispatch_v2.json` | Arch-keyed winners table (`by_arch.gfx942` / `by_arch.gfx950`). gfx942 = official baseline (empty winners, k32:false); fill in gfx942 winners as levers land. |
| `op_tests/flydsl_tests/bench_jagged_dense_bmm_perf.py` | The canonical perf+correctness bench (times ONE resolved config; no sweep). |
| `op_tests/flydsl_tests/tune_jdbba_*.py` | **Autotuning scripts** (build these): tier-A live-knob sweep (`xcd_c×xcd_w`) + tier-B module-constant sweep. Gate cos, time both regimes, emit winners → `by_arch.gfx942`. See *Autotuning scripts*. |
| `op_tests/flydsl_tests/test_jdbba_dispatch_v2.py` | Dispatch correctness test. |
| `aiter/ops/flydsl/gemm_tune/*` | In-repo tuner precedent (shape to mirror for the jdbba tuners). |
| `jagged_dense_bmm_mi300x_experiment_plan.md` | The neutral gfx942 lever plan (no prejudged outcomes). |
| `jagged_dense_bmm_broadcast_add_dev_journal.md` | Running methodology log. |

**Live per-call knobs today** (forwarded by the dispatch): `xcd_c`, `xcd_w`
(chiplet remap), `use_mfma_k32` (atom; gfx942 has NO 32-K atom → must be False
here), `block_k` (shape-derived: 128 if reduction K≤256 else 64).
**Module constants** in `jagged_dense_bmm_gen.py` (edit the source to sweep):
`BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=64`, `STAGES_A=2`, `THREADS=256`, and the
`tiled_mma` warp layout `(1,4,1)`.

---

## The lever menu (each a gfx942 hypothesis — no prejudged outcome)

Carried from the experiment plan; ordered roughly by expected leverage for this
**overhead-bound, memory-bound-on-A/Out** kernel. For each: implement on the
kernel, gate cos, measure both regimes, confirm structural ones in ISA, decide.

| # | Lever | What it does | gfx942 question |
|---|---|---|---|
| 1 | **MFMA atom** | bf16 tile size. gfx942 has **only 16×16×16** (no 32-K atom). | Confirm `use_mfma_k32=False` path is correct + is the floor. Not a choice here. |
| 2 | **BLOCK_K** | reduction-tile depth (128 small-K / 64 large-K). | Sweep per shape; balance barriers vs occupancy on gfx942's K-loop. |
| 3 | **BLOCK_M / BLOCK_N** | output tile size. Bigger = fewer/fatter tiles → fewer fills+epilogues (the amortization lever). | Sweep; **watch the 64 KB LDS ceiling** (2.5× smaller than gfx950 → may force smaller tiles). Try `BLOCK_N=N` to cover the whole output width in one block. |
| 4 | **Warp layout (m_warps/n_warps)** | distribution of the 4 warps over M/N (`tiled_mma`). | Sweep; the optimum is arch-specific. |
| 5 | **Pipeline STAGES** | software-pipeline depth over the K-loop. | 2 vs 3+; bounded by the 64 KB LDS limit and whether gfx942 is latency- or bw-bound. |
| 6 | **XCD chiplet remap** | clusters a group's M-tiles onto one XCD for Dense[b] L2 reuse (knobs C, W). | **Re-derive sign+magnitude in BOTH regimes** on gfx942's 8 XCDs. Do NOT import the gfx950 gate. Cold vs hot L2 may flip it. |
| 7 | **Epilogue C store** | fp32 acc → global C (the LDS C-shuffle wide-store already shipped, +6-17% on gfx950). | Verify the wide-store epilogue is correct + a win on gfx942's store path; try a direct/register epilogue at small N. |
| 8 | **Persistent scheduler** | on-device CUM prefix + occupied-tile-only traversal (skew only). | Compare vs static-grid+early-exit on skew per shape; route only where it wins on gfx942. |
| 9 | **A staging path** | global→LDS→reg vs global→reg / async copy. | Which staging the gfx942 memory pipe prefers. |
| 10 | **waves_per_eu** | occupancy hint. | Sweep 1-4; watch VGPR spills (pressure differs from gfx950). |
| 11 | **i64 offset math** | row-base offset in 64-bit before stride multiply. | **Correctness invariant — KEEP.** `seq_start·K` overflows i32 at large L (B1024_D512: 7.86M·512 > 2³¹). |
| 12 | **B-stationary multi-M-tile** | load Dense[b] once, run several M-tiles back-to-back so the pipeline never drains (the big amortization lever). | Port `small_m_hgemm`'s `PERSISTENT_N_TILES` to the **M** axis + `B_TO_LDS=True`. New kernel surface; high potential on the short-K regime. |

Lever #11 is a keep-always invariant. #1 is fixed by hardware. Everything else is
an open question to be answered with gfx942 numbers.

### The two regime-gated decisions to re-derive on gfx942
The dispatch makes two choices whose gates were derived on **gfx950** and must be
re-measured on gfx942:
1. **XCD remap on/off per regime** (lever #6) — remap trades Dense[b] L2 reuse vs
   chiplet load balance; both sides depend on XCD count + L2 behavior. Set the
   `uniform_seqlen`-gated default from gfx942 data.
2. **Persistent vs static under skew** (lever #8) — route to persistent only where
   it actually wins on gfx942.

---

## Autotuning scripts (how each grid-search lever is run)

The bench (`bench_jagged_dense_bmm_perf.py`) only *times one resolved config*; it
has no sweep. Autotuning is done by **dedicated tuner scripts** that drive the
grid and write winners. Build them under `op_tests/flydsl_tests/` and commit them
(the existing `aiter/ops/flydsl/gemm_tune/*` tuners are the in-repo precedent for
shape; mirror it).

A tuner must, for each of the 4 headline shapes × both regimes:
1. Enumerate the lever's grid (e.g. `xcd_c ∈ {1,16,32,60,120,240} × xcd_w ∈ {4,8}`).
2. **Correctness-gate every point** (`cos ≥ 0.999` vs torch eager) — skip, never
   record, a failing or crashing config.
3. Time with `triton.testing.do_bench` (cold-L2, the headline). For
   L2-reuse-sensitive levers (XCD remap, B-stationary) **also** take a
   `rocprofv3 --kernel-trace` p10 (hot-L2) — cold vs hot can flip the winner.
4. Pick the min-ms point per (shape, regime); print a TSV row per config and the
   chosen winner.
5. **Emit, don't silently hand-edit:** write the winners into a candidate JSON the
   loop reviews, then promote into `by_arch.gfx942.winners` of
   `jagged_dense_bmm_dispatch_v2.json` with a `source` string recording the sweep
   (grid, timing tool, `Mi`, date) — exactly the provenance the `gfx950` section
   already carries.

Two tiers, matched to what the dispatch can forward:

| Tier | Knobs | How the tuner sets them | Status |
|---|---|---|---|
| **A. Live-knob tuner** (do first) | `xcd_c`, `xcd_w` (and `use_mfma_k32=False`, `block_k`) | Pass directly to `jagged_dense_bmm_dispatched(..., xcd_c=, xcd_w=)` / `resolve_config` — **no kernel edit, no recompile churn beyond the per-config memoize.** | Runnable today; this is the XCD-remap re-derivation (lever #6) and the cheapest win. |
| **B. Module-constant tuner** | `BLOCK_M/N/K`, `STAGES_A`, `tiled_mma` warp layout, `waves_per_eu` | These are module constants / not forwarded by the dispatch yet (see *Files & knobs*). The tuner must **rewrite the constant (or parametrize `_build_launcher`) per grid point**, then reload — or the warp/wpe kernel clones must land first so the knob becomes a real argument. | Needs source edits / the missing clones before it does anything but recompile the same kernel. |

Keep tier-A and tier-B tuners as **separate scripts** (e.g.
`tune_jdbba_xcd.py`, `tune_jdbba_tiles.py`) so the cheap, always-valid XCD sweep
isn't blocked on the kernel-surface work. Every tuner honors the loop's keep/
revert discipline: a config is only a "winner" after it both passes cos and beats
the baseline ms it's compared against.

---

## Bound check (do this once, before levers)

Compute the floor so you optimize the right thing:
- **HBM-traffic floor:** bytes = `(L·D + B·D·N + B·N + L·N)·2` (A read-once +
  Dense + bias + Out write-once), `L = Σ M_b`. Divide by MI300X HBM BW (~5.3 TB/s
  peak) for the bandwidth-bound floor in ms. The bench already computes this `mem`
  term for `--metric bandwidth`.
- **Occupancy ceiling:** VGPR/AGPR + **64 KB LDS** limited (gfx942 LDS is
  2.5× smaller than gfx950 — recheck `smem_bytes` against 64 KB after any tile-size
  lever).
- If the kernel is near the HBM floor, lever #6/#3/#12 (traffic + amortization)
  matter most and compute-scheduling levers won't help.

## ISA check (for structural levers)

Dump the compiled HSACO and count per-iteration mnemonics:
`llvm-objdump -d --mcpu=gfx942 <hsaco>`. A faster variant shows fewer
`buffer_store_short` (scalar epilogue → the C-shuffle removes these), no
`ds_write` spill, no AGPR↔VGPR shuffles, fewer `s_waitcnt`. **A lever that doesn't
move this histogram changed nothing.** Use `rocprofv3` PMC (`MfmaUtil`,
`VALUBusy`, `LDSBankConflict`, `MemUnitStalled`, `MeanOccupancyPerActiveCU`) to
classify the bottleneck before choosing the next lever.

---

## Results table (keep this updated; do not commit it to git)

Mirror the autoresearch `results.tsv` discipline. Suggested columns (TSV):

```
commit	shape	regime	flydsl_ms	triton_ms	ratio	cos	status	lever
```

`ratio` = triton_ms / flydsl_ms (>1 means FlyDSL faster; 2.0 is the target).
`status` ∈ {keep, discard, crash}. One row per shape×regime per lever. The
deliverable is this table filled with **measured gfx942 data only**, the two
re-derived regime gates, and the final FlyDSL-vs-Triton standing on the 4 shapes
in both regimes.

---

## What does NOT help here (don't waste loop iterations)

- **Split-K / k_batch>1** — the *opposite* of amortization on this short-K kernel;
  shortens each block's K-loop and multiplies block count, each re-paying fixed
  cost. Keep `k_batch=1`.
- **Lengthening the reduction** — D is fixed by the problem.
- **Huge BLOCK_K** — kills the steady state that hides fill latency. Sweet spot is
  64-128, a tuning knob not a free win.
- **FP8 / bandwidth tricks when `MemUnitStalled`≈0** — adds VALU, which is then the
  bottleneck (cdna-kernel-opt rule).
- **The gfx950 dispatch winners** — wrong arch; they are the *starting hypothesis
  to re-test*, not answers.
- **`use_mfma_k32=True` on gfx942** — there is no 32-K atom; it must be False.

## Honest ceiling

Amortization levers recover the **overhead fraction** (~20-40% with a 4-8-step
loop), not a multiplier on peak flops. The realistic 2x prize comes from stacking:
a warm pipeline (B-stationary #12) + fatter tiles (#3) + resident B/bias + the
right XCD gate (#6) + the wide-store epilogue (#7) — and from the fact that the
Triton baseline itself is leaving ~20% on the table on MI300X. **Profile to
confirm the premise before investing in any one lever** (rocprofv3: MFMA-active vs
total cycles).

---

## Related skills & references

- **cdna-kernel-opt** — the general CDNA lever method (diagnosis, pipeline
  overlap, atom/layout matching, LDS swizzle, epilogue). This skill is its
  jdbba-specialized instance.
- **chiplet-xcd-remap** — lever #6 in depth (the C/W knobs, the L2-greedy trap).
- **autoresearch** (`~/autoresearch/program.md`) — the never-stop loop discipline.
- **Triton reference:** `generative_recommenders/ops/triton/triton_jagged.py`
  (`jagged_dense_bmm_broadcast_add_kernel`); walkthrough in
  `jagged_dense_bmm_triton_kernel_walkthrough.md`.
- **Design + journal:** `jagged_dense_bmm_broadcast_add_sami_plan.md`,
  `jagged_dense_bmm_broadcast_add_dev_journal.md`,
  `jagged_dense_bmm_mi300x_experiment_plan.md`.

## One-sentence takeaway

> Run the autoresearch keep/revert loop over the cdna-kernel-opt lever menu,
> scoped to jdbba on gfx942: baseline first, one lever at a time, correctness
> (cos≥0.999) before speed, both regimes measured, structural wins confirmed in
> the ISA histogram — because the gfx942 dispatch is an untuned heuristic and the 2x
> lives in amortizing this short-K kernel's fixed costs on MI300X, not in any
> single knob.
