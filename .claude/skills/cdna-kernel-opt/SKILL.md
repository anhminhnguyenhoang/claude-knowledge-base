---
name: cdna-kernel-opt
description: End-to-end method for optimizing HIP/MFMA kernels on AMD CDNA3/4 GPUs (MI300/MI325/MI350/MI355X, gfx942/gfx950). Use when writing, profiling, or speeding up a GEMM/attention/conv/reduction kernel on these chips, closing a gap to a faster reference (Triton/rocBLAS/CUTLASS), or deciding which lever to pull. Covers diagnosis, pipeline overlap, MFMA atom/layout matching, LDS swizzle, chiplet remap, and the ISA-histogram method. For the chiplet remap alone, use chiplet-xcd-remap.
argument-hint: [kernel file, op type, or perf gap]
---

# CDNA Kernel Optimization Method

A disciplined loop for making MFMA kernels fast on AMD CDNA3/4 (gfx942 MI300X,
gfx950 MI350X/MI355X). Distilled from the HipKittens paper (arXiv 2511.08083)
and the CK DSL Optimization Runbook — two independent sources that converge on
the same load-bearing claims.

**Governing principle:** Optimization is a measured loop, not a bag of tricks.
Classify the bottleneck with a cheap probe *before* changing anything, change
**one lever** at a time from the menu below, and confirm the change in the
**per-iteration ISA instruction count** — not just the latency number. The cost
you remove rarely disappears; it usually moves somewhere else.

## The loop

1. **Define** the op: shapes, layouts, dtypes, tolerances, boundaries.
2. **Baseline**: correctness gate + a perf reference (rocBLAS / Triton / CUTLASS).
3. **Classify the bottleneck** with a static probe + a short rocprof PMC pass.
4. **Read the reference's source** before sweeping knobs — the gap is usually
   *structural* (how the loop body is realized), not a launch-config knob.
5. **Pull one lever**, re-measure, diff the per-iter ISA histogram.
6. Repeat. Stop when you hit the reference or the remaining gap is structural
   and not knob-flippable.

## Step 1 — Diagnose before you change

Profile first. On a host with `rocprof-trace-decoder` use ATT; otherwise the
equivalent comes from one `rocprofv3` PMC pass + static ISA:

- **PMC counters**: `MfmaUtil`, `VALUBusy`, `LDSBankConflict`, `MemUnitStalled`,
  `MeanOccupancyPerActiveCU`.
- **Static ISA**: `llvm-objdump -d --mcpu=gfx950` on the dumped HSACO (count
  per-iter mnemonics).
- **Occupancy/registers**: VGPR / AGPR / SGPR / LDS from the kernel-stats header.

Read the signatures:

```
"compute-throttled by VALU/LDS plumbing, not HBM-bound"
    MfmaUtil low · VALUBusy dominant · LDSBankConflict >~5% · MemUnitStalled <1%
    → fix LDS/AGPR plumbing, NOT launch knobs. Demote all bandwidth levers.

"intermediate tile round-trips through LDS"
    ds_write_b<N>/iter ≈ tile_bytes/threads · s_barrier in inner loop
    → move the intermediate into registers

"MFMA atom lane-layout mismatch with next atom in chain"
    (v_accvgpr_read + v_accvgpr_write)/MFMA elevated · ds_swizzle/ds_bpermute in loop
    → switch to an atom whose C-out matches the next A-in
```

**Bandwidth is usually NOT the bottleneck on these kernels.** If
`MemUnitStalled` is near zero, drop every bandwidth-saving idea (FP8 KV storage,
bigger tiles for reuse) immediately — they add VALU work, which is the real
bottleneck.

## Step 2 — The levers (in rough priority order)

### A. Pipeline overlap — keep both engines busy
Every kernel is a race to keep the **MFMA pipe** and the **memory pipe** busy
simultaneously. The default schedule is **ping-pong / double-buffer**: while one
wave group computes, the other prefetches the next tile, then they swap.

- Double-buffer LDS (two halves); prefetch next tile while computing current.
- Place `s_waitcnt` as late as safely possible; put waits just before consumers.
- Prefer direct global→LDS async copy over register staging.
- Interleave MFMA / `ds_read` / global loads — avoid long memory runs followed
  by long MFMA runs.
- **Issue order matters**: issuing a dependent load *earlier* lets it overlap
  more compute (the "early-V" schedule in attention overlaps V with both QK and
  softmax, not just softmax) — but only when compute is ahead of memory.

CK knobs: `SoftwarePipeline.run_ping_pong`, `pipeline="compv4"` (double-buffer).
HK schedules: 8-wave ping-pong (default, balanced kernels); 4-wave interleave
(1 wave/SIMD, 512 regs, for register-heavy / mixed-shape backward passes —
~3× the code for ~20–30% more perf).

### B. MFMA atom shape & lane-layout matching
The chained-atom layout match matters **as much as** the atom's throughput.

- The **16×16×32** atom's C-output lane layout does **not** match its A-input,
  so a QK→PV chain needs an LDS round-trip or ~288 cross-lane permutes/iter to
  re-pack between atoms.
- The **32×32×16** atom's C-output **natively matches** the A-input → the same
  chain runs with **zero re-pack**. Prefer it when the chain pattern allows,
  even though both atoms have identical compute throughput.
- Default register tiles to the smallest atom (16×16×32) for max scheduling
  freedom; opt up to 32×32 explicitly for chained patterns.
- Use larger-K atoms (`16x16x32` over `16x16x16`, `32x32x16` over `32x32x8`)
  when K-packing is valid and within tolerance.

CK knobs: `use_mfma_32x32`, `use_transposed_qk_32x32`. Confirm the emitted
intrinsic with `probe_intrinsic_counts.py` / `count_instructions.py`.

### C. Intermediate residency — keep it in registers
If a tensor (P in attention, A/B in GEMM) is written to LDS then immediately
read back, that LDS round-trip is structural overhead. Move it into registers.
**Caveat:** register-residency alone is often a *net loss* — it can replace a
`ds_write` stripe with a worse cross-lane permute storm. It only wins when
**co-evolved** with the atom-shape and transposed-reader changes. A single lever
rarely is the whole change.

### D. LDS swizzle — match it to how data actually arrives
Bank conflicts can't be solved abstractly. On AMD, async DRAM→LDS writes are
**lane-contiguous**; handing the intrinsic an arbitrary swizzled LDS pointer
*compiles but corrupts output*. Express swizzle in the **address arithmetic**,
not by reshaping the destination pointer. XOR/cyclic swizzle only pays off when
paired with the matching async load distribution.

Per-arch choice (capacity-driven):

| Arch | LDS/CU | Preferred swizzle |
|---|---|---|
| gfx90a / gfx942 | 64 KB | **XOR** (capacity precious; 0 LDS overhead, +ALU) |
| gfx950 (MI350/355X) | 160 KB | **padding** (capacity abundant; simple offset, low ALU) |

```python
if LDS_total < 96*1024:   use_xor_swizzle()
elif LDS_avail >= 128*1024: use_padding_swizzle()
else: benchmark_both()
```

Asymmetry on gfx950: `ds_write_b128` has a 32-dword conflict period but
`ds_read_b128` has 64-dword — reads and writes may need **different** swizzles.

### E. Chiplet (XCD) work reordering
On these 8-chiplet GPUs, a launch-time block-ID remap reuses the per-XCD L2 and
buys ~+19% bandwidth for free on large reuse-heavy kernels. **This has its own
detailed skill — use `chiplet-xcd-remap`.** Quick form: two knobs, `C` (group
consecutive blocks onto one XCD) and `W` (walk in 2D windows); tune for
bandwidth, never L2 hit rate alone (the L2-greedy trap starves the LLC and goes
slower than baseline). CK knob: `chiplet_swizzle=True`.

### F. Epilogue store path
Vectorizing the epilogue is often the single largest win for a kernel that
already has a good main loop. Replace scalar `buffer_store` with wide vector
stores (`fp16x8`) — measured ~2× on direct-conv with no other change. Inspect
the `vmem_store` bucket first; scalar-store kernels show up immediately. Use an
LDS/CShuffle epilogue *only* when it meaningfully improves global coalescing.

### G. Compiler flags — last, and rarely the answer
Large gaps come from kernel **structure**, not flags (sub-1% on gfx950).
Validate correctness after every scheduler flag. **gfx950 trap**:
`s_sched_barrier` / `s_sched_group_barrier` are **silently removed** by the LLVM
backend — verify the `sched_barrier` bucket is 0; don't rely on them.
`-mllvm -enable-post-misched=0` can miscompile MLIR-generated kernels — risky.

## Step 3 — Confirm with the ISA histogram, not the stopwatch

The **per-iteration mnemonic count** from `llvm-objdump` is the unit of design
comparison. A faster reference typically shows: half as many (larger-K) MFMA,
no `ds_write` spill, no AGPR↔VGPR shuffles, far fewer `ds_bpermute` / `s_waitcnt`.

> **A change that doesn't move the per-iter ISA composition almost certainly
> didn't change anything** (the compiler may have already DCE'd it).

## Things that commonly do NOT help (and why)

| Attempted | Why it fails |
|---|---|
| FP8 KV storage when `MemUnitStalled`≈0 | HBM wasn't the bottleneck; FP8 dequant adds VALU, which *was* |
| Doubling tile size past default | Violates per-CU LDS ceiling, drops occupancy below 2 WG/CU |
| `num_warps` sweep at reference's pick | Amplifies LDS/AGPR conflicts without fixing them |
| Smoke-set-only validation | A "win" on a smoke set can regress the full cohort — always run the full cohort |
| L2-greedy chiplet config | Starves LLC; slower than baseline (see chiplet-xcd-remap) |
| Compiler hint sweep | Sub-1% movement; structure is the lever |

## Transferable principles

1. A static probe is faster than a sweep — it can disprove many knob rows at once.
2. Read the reference's source; don't infer its behavior from PMC alone.
3. Per-iter ISA histogram is the unit of design comparison; latency compresses too much.
4. One lever is rarely the whole change — co-evolve 2–3 when the first's
   apparent regression is an exposed downstream cost.
5. The dispatcher is a perf lever — no single variant wins every shape; per-shape
   branches close the long tail.
6. Validators encode past assumptions, not laws — re-examine restrictions when
   crossing a design boundary.
7. Things you remove are not free — check whether the removed cost reappears
   elsewhere (cross-lane permute, VGPR pressure, occupancy drop).

## One-sentence takeaway

> Diagnose the bottleneck cheaply, change one catalogued lever, and prove it in
> the per-iteration ISA histogram — because on CDNA the gap to a faster kernel is
> almost always *structure* (atom layout, LDS round-trips, schedule, chiplet
> mapping), not knobs or flags, and every cost you remove tends to move rather
> than vanish.

Sources: HipKittens (arXiv 2511.08083); CK DSL Optimization Runbook (§3, §6–§12, §17.4, §21).
Related skills: chiplet-xcd-remap (the XCD remap in depth), opus-kernel-best-practice (compile-time).
