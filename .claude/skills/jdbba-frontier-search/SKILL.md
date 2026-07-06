---
name: jdbba-frontier-search
description: Frontier optimization loop for the FlyDSL jagged_dense_bmm_broadcast_add (jdbba) kernel on AMD MI300X / gfx942, for use AFTER the knob-level menu is exhausted. Where jdbba-autoresearch tunes existing knobs one at a time (greedy) and has hit the gfx942 MFMA-issue-latency ceiling, this skill (1) adds NEW kernel-surface levers that attack that ceiling directly -- the 32x32x8 bf16 atom, dual-accumulator MFMA interleave, hybrid A-fragment register prefetch -- and (2) runs a JOINT co-optimization search over (atom x BLOCK_K x threads x B_STAGES x warp-split) to find a co-optimum the greedy path misses. Use when asked to push jdbba past its current best, find new performance uplift, break the MFMA ceiling, do a joint/multi-knob autotune, or "lock in a new win on the jagged bmm". Every run ratchets so that it ends with either a committed uplift or a documented dead-end that shrinks the frontier.
argument-hint: [a frontier lever R1-R4, a shape to focus, or nothing to run the ratchet]
---

# jdbba Frontier Search -- push past the knob ceiling on MI300X (gfx942)

A **frontier optimization loop** for the FlyDSL
`jagged_dense_bmm_broadcast_add` (jdbba) kernel on **AMD Instinct MI300X
(gfx942 / CDNA3)**, for the regime **after the one-knob-at-a-time menu is
spent**. It is the successor to `jdbba-autoresearch`, not a replacement:

- **jdbba-autoresearch** ran the greedy edit -> cos-gate -> measure -> keep/revert
  loop over the CDNA lever menu (BLOCK_K, XCD remap, threads, B_STAGES,
  warp-split). It reached a **gfx942-optimal config on the current kernel
  surface** and diagnosed the remaining ceiling as **MFMA-issue / dependency
  latency** (MfmaUtil pinned ~28-44% of peak, SQ_WAIT_INST_ANY ~54%,
  MemUnitStalled ~0.1%, occupancy-doubling inert).
- **This skill** attacks that ceiling with **two things the greedy loop could
  not do**:
  1. **New kernel-surface levers (R1-R3)** that change the MFMA issue cadence
     itself, not just the knobs around it.
  2. **A joint search (R4)** over interacting knobs, because once the atom
     becomes a real knob the optimum is a *combination*, and greedy provably
     misses co-optima.

**Goal:** lock in **new measured uplift over the current committed best** on the
four headline shapes in both regimes -- ratcheting the FlyDSL/Triton ratio down
each run. The honest 2x blocker is hardware (gfx942 lacks the K=32 bf16 atom);
this skill chases the recoverable fraction between today's config and that wall.

---

## Inherit everything operational from jdbba-autoresearch

**Do not duplicate the shared machinery -- read it there and follow it exactly.**
This skill assumes you have the parent skill's sections in force:

- **Environment** -- the `jdbba-flydsl` docker container, the `/workspaces/meta/aiter`
  bind mount, the canonical bench invocations, the stale-aiter-JIT-build fix, the
  arch-keyed dispatch JSON. (jdbba-autoresearch -> "Environment".)
- **The metric and the rules** -- `do_bench` cold-L2 ms as the headline,
  ratio = flydsl_ms / triton_ms, **correctness gate FIRST** (cos >= 0.999 AND
  mean-signed-error on all 4 shapes + edge cases), **both regimes measured**,
  confirm structural levers in the ISA histogram. (jdbba-autoresearch ->
  "The metric and the rules".)
- **The op spec, the (B,D,K,N) naming trap, the headline shapes, the shape
  facts.** (jdbba-autoresearch -> "The op" and "Files & knobs".)
- **Bound check / ISA check / PMC classification** procedures. (jdbba-autoresearch
  -> those sections.)
- **The measurement discipline** -- autotune-min-not-median for Triton, cold vs
  hot L2 brackets, change-one-thing, re-measure surprises. (KB
  `flydsl-jdbba/20-methodology/measurement-methodology.md`.)

If any of those has drifted, the parent skill is the source of truth for the
mechanics. This file only adds the **frontier levers and the joint search**.

**Written-output convention (user memory): ASCII only.** No em dashes, arrows,
multiplication signs, etc. -- use `--`, `->`, `x`. Applies to the results doc,
commit messages, and the dispatch JSON `source` strings you emit.

---

## Precondition: you are actually at the frontier (do not skip)

This skill is wasteful if run before the greedy menu is exhausted. Confirm the
handoff state BEFORE pulling a frontier lever:

1. **Re-establish the current committed baseline.** Run the canonical both-regime
   bench on `HEAD` of `anguyenh/flydsl-*`. This is the number every frontier lever
   must beat -- NOT the original `fc14a4e01` heuristic baseline. Record it as the
   ratchet's current rung (see *The ratchet*).
2. **Confirm the ceiling diagnosis still holds** with a fresh PMC pass on the
   worst-ratio shape (usually a D512 uniform): MfmaUtil, SQ_WAIT_INST_ANY,
   SQ_INSTS_VALU vs SQ_INSTS_MFMA, MemUnitStalled, MeanOccupancyPerActiveCU. You
   are looking to re-confirm: **MFMA-issue-latency bound, not occupancy- or
   memory-bound.** If the profile has changed (e.g. a new lever shifted the
   bottleneck to memory), the frontier priorities below reorder -- say so.
3. **Check the greedy loop has nothing cheap left.** If jdbba-autoresearch's
   discard table has an untried knob point, run that first -- it is cheaper than a
   kernel-surface change. Only proceed to R1-R4 once the knob menu is genuinely
   spent.

If you cannot confirm (1)-(3), fall back to `jdbba-autoresearch` and come back.

---

## The frontier lever menu (each attacks the MFMA-issue-latency ceiling)

Carried from `jagged_dense_bmm_mi300x_experiment_plan.md` section 5b (R1-R4) and
the dev-journal 2026-07-03 entry. Each is a **new kernel surface**, not a knob --
they require real edits to `jagged_dense_bmm_gen.py`'s MFMA/fragment/pipeline
code, so each is a multi-hour experiment, not a sweep point. Ordered by leverage.

| # | Lever | Mechanism | The gfx942 question | Cost / risk |
|---|-------|-----------|---------------------|-------------|
| **R1** | **32x32x8 bf16 MFMA atom** (highest leverage) | gfx942 lacks `v_mfma_f32_16x16x32_bf16` (the K=32 atom, the hard 2x blocker) but HAS `v_mfma_f32_32x32x8_bf16`. A bigger output-tile atom issues **fewer MFMA instructions for the same tile** -> directly cuts the issue count that pins MfmaUtil. | Does the 32x32x8 fragment layout compile + stay cos>=0.999, and does the lower issue count lift MfmaUtil above ~28-44%? | 32x32 C-frag is **16 fp32/lane** (vs 4 for 16x16) -> big AGPR/VGPR jump; watch the occupancy cliff and the LDS C-shuffle epilogue (its N-repeat granularity assumes 16x16). New fragment wiring. |
| **R2** | **Dual-accumulator MFMA interleave (ILP)** | Keep **two independent C-accumulator tiles** and interleave their MFMA issue so chain A's MFMA issues while chain B waits on its s2r LDS read -- hides issue latency **without more waves** (PMC proved waves inert). | Does interleaving two dependency chains lift MfmaUtil / drop SQ_WAIT_INST_ANY (54%)? | Two accumulators ~= 2x AGPR pressure; may collide with the occupancy the kernel already needs. The classic software-pipelining-for-ILP restructure. |
| **R3** | **Hybrid A-fragment register prefetch** | The B-prefetch win (+5-12%) staged B in registers 2-ahead. A still goes LDS with a **1-ahead s2r read sitting directly in the MFMA critical path**. Prefetch A's s2r fragments deeper into registers to move LDS-read latency off the chain. | What **hybrid** LDS/register A-staging *depth* hides the read without blowing VGPR? (Full global->reg is already refuted -- VGPR wall. So the question is depth, not all-or-nothing.) | VGPR is the binding resource; this trades LDS-read latency for VGPR. Sweet-spot depth only. |
| **R4** | **Joint / multi-knob co-optimization** | The greedy loop tunes one knob at a time and takes the local best at each step. A **joint search over `(atom x BLOCK_K x threads x B_STAGES x warp-split)`** can find a co-optimum greedy misses -- especially once R1 makes `atom` a real knob that reshapes the BLOCK_K / warp-split trade. | Does a joint sweep beat the greedy per-shape winners on any shape/regime? | Search cost (grid size x JIT compile). Managed by the staged search below, not brute force. |

**Do NOT re-open** fp8 / split-K / bigger-tiles (BLOCK_M/N 256) / fusion /
warp-specialized rewrite -- all refuted by PMC in the parent campaign (the
warp-spec ceiling is hardware, barrier already free, VALU only 20% busy). Re-check
the parent's discard table before proposing anything; a frontier lever must be
genuinely new.

### Why R1 first
R1 is the closest *software* proxy for the missing K=32 atom -- the one thing the
profile says would actually move the ceiling. It is also the enabler for R4: until
`atom` is a real, correct knob, the joint search has only the old axes. Land R1
(even as a per-shape opt-in), then let R4 co-optimize around it.

---

## The ratchet (this skill's core discipline -- lock in uplift every run)

The parent loop's rule is "keep if better, revert if not." The frontier adds a
**monotonic ratchet**: the committed best only ever moves **down** (faster), and
**every run must end in one of two states**, both of which advance the search:

- **RUNG CLIMBED** -- a frontier lever (or a joint-search config) passed cos AND
  beat the current committed best ms on at least one shape/regime with no
  regression elsewhere it is promoted for. **Commit it**, update the dispatch JSON
  winner + `source`, update the results doc's "current best" table, and record the
  new rung. The next run must beat *this*.
- **FRONTIER NARROWED** -- no lever beat the best, but the run produced a
  **documented, measured dead-end** (with the PMC/ISA reason it failed) that
  removes a region from the search. This is a real deliverable: it stops the loop
  and every future loop from re-trying that region. Add it to the frontier
  dead-end table.

A run that ends with neither -- no commit, no new documented dead-end -- is a
**wasted run**; do not let that happen. If a lever is inconclusive (within noise),
that itself is the finding: record "R_n inconclusive within +-X% at depth D,
region closed unless the bottleneck shifts."

**Never regress the rung.** Before committing any frontier change, re-run the full
both-regime bench and confirm no headline shape got slower than the committed
best. A frontier lever that helps D512 but regresses D256 is promoted **per-shape
only** (via the dispatch JSON winner for that shape), never globally.

---

## The loop (frontier, ratcheted)

Work on the `anguyenh/flydsl-*` line. **LOOP until a rung is climbed or the batch
is exhausted:**

1. **Precondition check** (above) -- current committed baseline re-measured, ceiling
   re-confirmed, greedy menu confirmed spent.
2. **Pick ONE frontier lever** R1-R3 (a kernel-surface change) OR launch the R4
   joint search (below). Prefer R1 if the atom is not yet a knob.
3. **Implement it in isolation** on a clone/worktree of `jagged_dense_bmm_gen.py`
   (never the production tree). Keep the change to the one mechanism.
4. **Correctness gate FIRST** -- bench `-test`, cos >= 0.999 AND mean-signed-error
   on all 4 shapes + edge cases (empty group, partial bottom tile,
   one-long-many-short). A new fragment layout (R1) or reordered issue (R2) is a
   prime spot for a K-reduction mis-accumulation that cosine alone masks -- use
   both checks. Fix or revert before timing.
5. **Confirm the mechanism moved in the ISA + PMC**, not just the stopwatch. R1:
   fewer MFMA instructions in `llvm-objdump`; MfmaUtil up. R2: SQ_WAIT_INST_ANY
   down. R3: the s2r read off the critical path. **A frontier lever whose target
   counter did not move did nothing -- treat as a dead-end, not a maybe.**
6. **Measure both regimes** with `do_bench` (cold-L2). For anything L2-sensitive
   also take the rocprofv3 p10 hot-L2 reading; cold vs hot can flip the verdict.
7. **Ratchet decision** (above): RUNG CLIMBED -> commit + promote + record;
   FRONTIER NARROWED -> record the measured dead-end with its counter reason. Then
   go to 2 with the next lever, or fold (below) if you have >1 winner.

**Never stop to ask "should I keep going?"** -- pull the next frontier lever or
launch the joint search. Out of R1-R4 ideas: re-profile the *new* current best (a
landed lever shifts the bottleneck -- the next lever follows the *new* dominant
counter), or hand R4 the newly-real knob.

---

## R4: the joint co-optimization search (the anti-greedy engine)

The greedy loop's blind spot: it fixes BLOCK_K, then tunes threads, then
warp-split -- taking the local best at each step. But these **interact** (atom
reshapes the warp-split optimum; BLOCK_K and B_STAGES both spend the pipeline;
threads sets the warp count that warp-split partitions). The joint optimum can sit
at a point **no greedy path visits**. R4 searches the product space directly.

**The joint axes** (all real `_build_launcher` arguments after R1 lands the atom
as a knob):

| Axis | Values (starting grid) | Notes |
|------|------------------------|-------|
| `atom` | {16x16x16, 32x32x8} | 32x32x8 only after R1 lands + passes cos. This is the new axis. |
| `BLOCK_K` | {64, 128} | 128 refuted *alone* on D512, but may compose differently with 32x32x8 (fewer, fatter MFMA). |
| `THREADS` | {256, 512} | warp count = THREADS/64. |
| `B_STAGES` | {2, 3, 4} | 3 is the greedy winner; 4 within noise, 5 regressed -- but atom changes the VGPR budget. |
| `warp-split` (m_warps, n_warps) | derived from THREADS + BLOCK_N, or explicit {(1,4),(2,2),(2,4)} | (2,2) compile-failed against the C-shuffle epilogue in the greedy run -- if R1/R2 rewrites the epilogue, re-open it. |

**Do NOT brute-force the full product** (2x2x2x3x~3 = 72 points x 4 shapes x 2
regimes x JIT compile = infeasible and mostly illegal-LDS/occupancy-dead points).
Search it in stages:

1. **Prune by static legality first.** For each candidate tuple compute
   `smem_bytes` (A double-buffer = `BLOCK_M*BLOCK_K*STAGES_A*2`) and the C-frag
   VGPR/AGPR footprint (16x16 -> 4 fp32/lane; 32x32 -> 16). Drop every tuple that
   busts the **64 KB LDS** ceiling or the occupancy-1-wave cliff **before**
   compiling. This kills most of the grid on paper.
2. **Coordinate-descent seeded from the greedy winner, but with restarts.** Start
   at the current committed config. Sweep one axis fully, move to its best, sweep
   the next -- but **re-sweep the first axis after the atom changes** (the whole
   point: the atom move invalidates the earlier local optima). Do >=2 full passes
   until no axis wants to move. This catches most co-optima at a fraction of full
   grid cost.
3. **Confirm the co-optimum against a random-restart spot check.** Sample a
   handful of legal tuples far from the descent path; if one beats the descent
   winner, the surface is multi-modal -- widen to a fuller grid on the axes that
   disagreed. If none beats it, promote the descent winner.
4. **Every point is cos-gated and both-regime timed** (the loop's rules do not
   relax for a sweep). A tuple is a "winner" only after cos passes AND it beats the
   ratchet's current rung.

**Build R4 on the existing tuner scaffolds** -- `tune_jdbba_tiles.py` (tier-B
module-constant sweep) and `tune_jdbba_xcd.py` (tier-A live-knob) already
enumerate a grid, cos-gate, time both regimes, and emit winners. R4 is a
**joint** driver over those: parametrize `_build_launcher` per tuple, reuse their
cos-gate + do_bench + winner-emit code, add the legality prune (step 1) and the
coordinate-descent-with-restart controller (step 2). Commit it as
`tune_jdbba_joint.py` next to them; emit winners into a candidate JSON the loop
reviews before promoting into `by_arch.gfx942.winners` with a full `source`
provenance string (grid, tool, Mi, date), exactly like the tier-A section already
carries.

---

## Fan-out and fold (parallelize the frontier levers)

Frontier levers are independent experiments -- run them concurrently, serialize
only the decision. Follow the parent skill's **"Fan-out: parallelize the levers,
then fold the winners"** section exactly, with these frontier specifics:

- **Up to 4 mutually-exclusive attempts per batch**, each a *different* frontier
  lever (R1 / R2 / R3) or *different points* of R4's grid that need source edits.
  Two subagents must never edit the same kernel region.
- **Isolation is mandatory** -- each subagent gets its own git worktree
  (`isolation: "worktree"`); the shared GPU serializes `do_bench`, but the source
  trees must be separate.
- **Each subagent runs the full frontier mini-loop** on its one attempt: implement
  -> cos+mean-error gate (4 shapes + edges) -> ISA/PMC confirm the target counter
  moved -> both-regime do_bench (+ hot-L2 if L2-sensitive) -> structured verdict.
- **Structured verdict, not prose:** `{lever, files_touched, per-shape x regime ms
  + cos + mean_err, ratio vs current-rung, target-counter delta (MfmaUtil /
  SQ_WAIT_INST_ANY / etc), keep/discard, reason}`. Discards come back with the
  counter reason so the frontier dead-end table grows.
- **Fold the winners** (parent skill's fold rules): partition by compatibility.
  R1 (atom) + R3 (A prefetch depth) may compose; R1 + R2 (dual-accumulator)
  **both** grow AGPR and likely fight for occupancy -- do not stack blindly, check
  the combined VGPR/AGPR + `smem_bytes` against the ceiling, then **apply
  compatible winners on a fresh clone and RE-MEASURE**. The stacked number is not
  the product of the individual ones -- interactions (AGPR pressure from two
  accumulators, occupancy from a fatter atom) can erase a win. Only the re-measured
  stacked number counts, and it must beat the rung to promote.

**Stop condition for the batch:** a stacked/single config climbs the rung
(commit + fan out the next batch against the new rung), OR a full batch of 4
returns no rung-climb -- then you have >=4 new documented dead-ends (frontier
narrowed), re-profile the current best to find the new dominant counter, and pick
the next batch from *that* diagnosis.

---

## Frontier dead-end table (append every narrowing; keep in the results doc)

The parent skill has a knob-level discard table. This skill maintains a
**frontier** one -- kernel-surface / joint-search regions proven dead, each with
the counter that killed it, so no future run (or subagent) re-tries it:

```
lever/region        config                         result              killed_by (counter)
R1 32x32x8 atom     <shape>, <depth>               <ms vs rung / cos>  <MfmaUtil delta, AGPR/occupancy>
R2 dual-accum       <shape>                        <ms vs rung>        <SQ_WAIT_INST_ANY delta, AGPR>
R3 A-reg-prefetch   depth=<D>                      <ms vs rung>        <VGPR, s2r-on-critical-path?>
R4 joint            <tuple>                        <ms vs rung / cos>  <legality / local-optimum note>
```

Seed it from the parent campaign's already-known frontier facts: the K=32 atom is
a *hardware* absence (not a lever); the warp-specialized rewrite ceiling is
hardware (barrier already free, VALU 20% busy); BLOCK_M/N=256, STAGES_A=3,
global->reg A, wpe>1 are all knob-level dead (see parent discard table -- do not
re-run them here).

---

## Results doc (extend the existing one; do not commit to git)

Extend `jdbba-autoresearch-results.md` -- do not start a new file. Add:

- A **"Frontier ratchet"** section: the current rung (committed best per shape x
  regime, its commit hash), and a rung-history log (each climb: lever, commit,
  ms before -> after, ratio, the counter that moved).
- The **frontier dead-end table** (above).
- For each landed frontier lever, a short writeup mirroring the parent's per-lever
  entries: what, mechanism, per-shape both-regime numbers, the ISA/PMC proof the
  target counter moved, and any FlyDSL tracing gotcha hit (e.g. constexpr-range
  unrolling, fragment-layout wiring).

The deliverable of a frontier run is: **the rung moved (with proof the MFMA-issue
counter moved to justify it), or the frontier table grew (with the counter that
closed the region).**

---

## Related skills and references

- **jdbba-autoresearch** -- the parent; the greedy knob loop + all shared
  environment/metric/rules/bound-check/ISA-check machinery this skill inherits.
  Run it first; run this after its menu is spent.
- **cdna-kernel-opt** -- the general CDNA lever method (diagnosis, atom/layout
  matching, pipeline overlap, epilogue). R1/R2/R3 are its atom / ILP /
  prefetch levers applied to jdbba's frontier.
- **chiplet-xcd-remap** -- lever #6 in the parent (already tuned on gfx942; a
  joint-search axis only if R1/R2 reshape it).
- **KB `flydsl-jdbba/`** -- `10-optimization-case-study/02-winning-levers.md` (the
  16x16x32 atom wiring, the C-shuffle epilogue that R1 must respect),
  `03-dead-ends.md` (the byte-floor reasons bigger tiles / B-in-LDS / async fail --
  do not re-run), `20-methodology/` (measurement + hot-loop-scheduling method).
- **Plan + journal:** `jagged_dense_bmm_mi300x_experiment_plan.md` section 5b
  (R1-R4 full mechanism), `jagged_dense_bmm_broadcast_add_dev_journal.md`
  (2026-07-03 frontier entry, the batch-5/6 PMC verdict this skill builds on).
- **autoresearch** (`~/autoresearch/program.md`) -- the never-stop loop discipline
  the ratchet specializes.

## One-sentence takeaway

> After the greedy knob menu is spent and the kernel is MFMA-issue-latency-bound
> on gfx942, run the ratcheted frontier loop: attack the issue ceiling with new
> kernel-surface levers (32x32x8 atom R1, dual-accumulator ILP R2, hybrid A-reg
> prefetch R3) and a joint co-optimization search (R4) over the interacting knobs
> greedy tuned separately -- cos-gated, both-regime measured, and every run must
> either climb the rung (commit a new best, proven by the MFMA counter moving) or
> narrow the frontier (a measured dead-end with the counter that killed it).
