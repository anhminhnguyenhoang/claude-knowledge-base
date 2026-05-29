---
name: unified-attention-2d
description: Runbook S17.4: multi-day UA-2D pass closing a Triton gap on gfx950. Profile-first (PMC fallback), read-the-reference, per-iter ISA histogram as design-diff, the structural-change ladder, what did NOT help, diagnostic signatures, pitfalls, and 8 transferable principles.
source: ck-dsl-optimization-runbook.md (lines 2119-2312)
---

### 17.4 Unified Attention 2D Optimization Pass

A multi-day case study of closing a substantial latency gap between
the `UnifiedAttention2DTiledSpec` kernel and a Triton reference across
a production-trace cohort of bf16 prefill-2D shapes on gfx950. The
methodology, levers, and pitfalls below are written to be transferable
to any kernel optimization that targets a known-better reference.

This is the worked example for the "profile first, change second"
methodology under §3, the LDS principles in §6.3 / §6.4, the
MFMA-atom selection in §7, the pipelining choices in §8, and the
dispatcher / autotuning discipline in §12.

#### Methodology lessons

**Profile first; PMC fallback when ATT decoder unavailable.** On a
host without `rocprof-trace-decoder`, the §3.1a ATT path is
unreachable. The equivalent diagnostic deliverables come from:

1. PMC counters in one `rocprofv3` pass:
   `MfmaUtil`, `VALUBusy`, `LDSBankConflict`, `MemUnitStalled`,
   `MeanOccupancyPerActiveCU`.
2. Static ISA from the dumped HSACO via `llvm-objdump -d --mcpu=gfx950`
   (use `probe_isa_inspect.py` or `count_instructions.py`).
3. Kernel-stats header from the same `rocprofv3` pass for VGPR / AGPR
   / SGPR / LDS / grid / workgroup (or `probe_occupancy.py` statically).

The worst-shape PMC reading on this kernel showed the classic
"compute-throttled by VALU/LDS plumbing" signature: single-digit
`MfmaUtil`, dominant `VALUBusy`, elevated `LDSBankConflict`, and
near-zero `MemUnitStalled`. HBM was definitively ruled out by the
last metric, so every bandwidth-saving lever (FP8 KV storage, larger
tiles for reuse) was demoted immediately.

**Read the reference implementation before knob-sweeping.** The
original plan had a long knob-sweep table (`num_warps`, `tile_size`,
`kv_storage_dtype`, etc.). PMC plus a side-by-side read of the
reference's source revealed that **both kernels were already running
with identical launch config** — same `BLOCK_M`, `TILE_SIZE`,
`num_warps`, `waves_per_eu`, and workgroup size. The gap was
structural — in **how the loop body was realized**, not in the
launch knobs. The original sweep would have moved nothing.

**HSACO per-iter mnemonic histogram is the design-diff gold standard.**
Once one source said "P stays in registers" and the other said
"P round-trips through LDS", the way to confirm was a per-iter
mnemonic count from `llvm-objdump`. The reference emitted half as
many (but larger-K) MFMA instructions, no `ds_write_b16` P-spill, no
AGPR↔VGPR shuffles around the accumulator, and a fraction of the
total K-loop instructions of the original DSL kernel.

The opcode delta — counted, not estimated — is what makes structural
differences explicit. Latency numbers compress too much.

#### Structural change ladder (lever family by lever family)

The pass that closed most of the gap stacked structural changes in
roughly this order. Each row is one lever; the cumulative effect is
the union, not the sum.

| Lever family | Concrete change | What it removes / unlocks | DSL knob (§12.1) |
|---|---|---|---|
| MFMA atom shape | switch QK / PV from the 16×16 atom to the 32×32 atom that natively aligns its C-output with its A-input | the cross-lane re-pack between QK and PV that the 16×16 chain needs | `use_mfma_32x32`, `use_transposed_qk_32x32` (§12.1.C) |
| Intermediate residency | keep softmax P in registers instead of spilling through LDS | the P-LDS allocation, the per-iter `ds_write_b16` stripe, and the in-loop `s_barrier` | `use_register_pv` (§12.1.K) — only wins when paired with the atom + reader changes below |
| LDS reader path | use the transposed-tile LDS reader (`ds_read_tr16_b{64,128}`) for V | the per-iter `v_readlane / v_writelane` register-side reshape | `use_transposed_pv_tr_read` (§12.1.K) |
| Lane ownership | each 32-lane half consumes only the K rows it already owns; matching half-local V reads | one cross-half reshape per iter | `use_transposed_half_local_pv` (§12.1.K) |
| Schedule | issue V async copy *before* QK so V overlaps both QK and softmax (not just softmax) | latency exposed when V was issued mid-loop | `use_early_v_schedule` (§12.1.D, §8.1) |
| Tile size matched to useful work | for sliding-window shapes, halve `tile_size` so each K-iter doesn't waste half the loop on masked-out columns | the wasted K-loop work, the wasted LDS reads, the wasted waits | `tile_size` (§12.1.B), `_select_2d_tile_size` (§12.1.O) |
| Dispatcher per-shape branch | one selector picks the early-V variant for no-sliding-window shapes, a different variant for sliding-window | the long-tail of shapes a single variant can't win | `_select_*` / `_enable_*` (§12.1.O) |

The structural insight at each step:

- **Atom shape**: the 32×32 atom is structurally cheaper *not*
  because it has higher arithmetic throughput — it has the same
  throughput as the 16×16×32 atom — but because its **C-output lane
  layout natively matches its A-input lane layout**, so the QK→PV
  chain skips the re-pack the 16×16 atom requires. (§7.2.)
- **Register-PV alone is a net loss** with the wrong atom: removing
  the P→LDS spill replaces a `ds_write_b16` stripe with a much
  larger cross-lane permute storm. The cost of a structural change
  is rarely visible until you look at the per-iter ISA composition.
  (§9.3 caveat.)
- **A structural change often needs multiple co-evolved levers** to
  actually win. The atom + register-residency + transposed reader
  trio works together; any subset of two loses to the original.
- **Cross-lane reshape can be sidestepped by reorganizing what each
  lane half owns**, not just by removing the reshape. (§5.3.)
- **Latency-hiding scheduling helps when compute is ahead of memory.**
  If the compute window is already saturated, issuing memory earlier
  just enlarges the LDS budget without helping. (§8.1.)
- **Tile size must match the useful work per tile**, not the LDS
  budget's maximum. For sparse masks (sliding window), a smaller
  tile cuts wasted work proportionally. (§13.3.)
- **The dispatcher is a perf lever.** A single kernel variant rarely
  wins every shape; per-shape branches close the long tail. (§12.3.)

#### Things that did NOT help (and why)

| Attempted lever | Outcome | Why it failed |
|---|---|---|
| FP8 KV storage (`kv_storage_dtype="fp8e4m3"`) | neutral / regression | `MemUnitStalled` was near zero; HBM was not the bottleneck. FP8 dequant adds VALU work, which **was** the bottleneck. |
| Doubling `tile_size` past the default | regression | Doubled LDS-per-WG, violated the per-CU LDS ceiling, dropped occupancy below 2 WGs/CU. |
| `num_warps` sweep | flat | Already at the reference's pick. Adding warps without fixing the LDS / AGPR plumbing only amplifies the conflicts. |
| Two KV tiles per accumulator update (`use_grouped_kv2_softmax`) | smoke set looked promising; full cohort regressed materially | Always run the full cohort, not a smoke set. |
| Specialized sliding-window prefill wrapper | flat or regression | Slower than the small-tile variant on every sliding-window shape it covered. |
| Force VGPR-form MFMA (`use_agpr_alloc_zero`) | flat | The chosen path already had zero AGPR moves; the lever was redundant. |
| Compiler hint sweep (`waves_per_eu`, `maxnreg`, scheduling modes) | sub-1 % movement | Compiler flags very rarely close large gaps (§10.2). |
| Register-P proxy variant (kept allocation removal without changing the read path) | within measurement noise | HSACO diff showed identical hot-loop ISA — the only "change" was removing an unused LDS allocation that the compiler had already dead-code-eliminated. **A change that doesn't move the per-iter ISA composition almost certainly didn't change anything.** |

#### Diagnostic signatures collected from this pass

These are the qualitative PMC + ISA patterns that mapped to each
finding. The thresholds are approximate — calibrate to your kernel
family before relying on them.

```text
"compute-throttled by VALU/LDS plumbing, not HBM-bound"
    MfmaUtil           well below the architecture's healthy range
    VALUBusy           dominant (most of the cycles)
    LDSBankConflict    above the action threshold (~5 % of LDS cycles)
    MemUnitStalled     negligible (<1 %)
    → fix the LDS / AGPR plumbing, not the launch knobs

"Intermediate tile round-trips through LDS"
    ds_write_b<N> per outer-iter ≈ (tile_bytes / threads)
    LDSBankConflict elevated for narrow stripe writes (b16 on
        wide-bank LDS)
    s_barrier present in the inner loop
    → consider register residency for the intermediate

"MFMA atom lane-layout mismatch with the next atom in the chain"
    (v_accvgpr_read + v_accvgpr_write) / MFMA elevated
    ds_swizzle_b32 or ds_bpermute_b32 in the inner loop
    cross-lane permute count per iter dominates
    → switch to an atom shape whose C-out matches the next A-in
```

#### Pitfalls and gotchas

- **Spec-validator restrictions can mask the actual win path.** The
  `attention_tiled_2d.py` validator initially raised on
  `use_register_pv` + `use_mfma_32x32` together; lifting that gate
  was the prerequisite to the winning combination. Validators added
  "to be safe" for a v1 limitation can outlive the limitation. Always
  ask whether the constraint reflects real numerics or just a
  previous caution.
- **HSACO module-cache aliasing.** When two kernel specializations
  share the same display symbol but differ in compile-time constants
  (e.g. `num_seqs`, binary-search trip count), the runtime can serve
  the wrong cached HSACO. Fix: salt the kernel symbol with a shape
  hash before compilation.
- **`REGS_PER_LANE`-dependent invariants must be audited when the
  MFMA atom changes.** Code that writes to a fixed slot index of a
  per-lane state tensor can be correct for one atom's
  `REGS_PER_LANE` and silently broken for another's. The atom switch
  in this pass surfaced an elevated `max_abs_diff` localised to
  specific query positions — caused by a sinks-init loop that
  covered only a fraction of the new per-lane state. Always re-test
  sinks / masks / initialisation paths when changing the atom.

#### What the remaining gap looks like

After the structural ladder above, the best kernel's K-loop still
emits, per outer iteration, far more AGPR↔VGPR moves, far more
`ds_bpermute_b32`, far more `s_waitcnt`, and several × the VALU ops
of the reference. These are **structural MFMA-pipeline +
softmax-realisation differences** — not knob-flippable. Closing them
requires a different lane layout for the QK output that natively
aligns with the PV A-operand (zero AGPR shuffle), plus a softmax
pattern that uses the cheap CDNA cross-lane VOP1 instead of the LDS
DMA path. This is exactly the "Closing the Last 5 %" pattern from
Case Study 5 in `utilities/skills/empirical-case-studies.md`.

#### Take-away principles (transferable to other kernels)

1. **A static probe is faster than a sweep.** A short PMC pass plus
   a `probe_isa_inspect.py` run can disprove several knob-sweep rows
   at once.
2. **Read the reference implementation's source.** It is faster than
   inferring from PMC alone what the reference is actually doing
   differently.
3. **Per-iter ISA histogram is the unit of design comparison.**
   Latency compresses too much; opcode counts make structural
   differences explicit.
4. **One lever is rarely the whole change.** Co-evolve two or three
   levers when the first lever's apparent regression is caused by
   an exposed downstream cost.
5. **The dispatcher is a perf lever.** A single kernel variant
   cannot win every shape; per-shape branches close the long tail.
6. **Validators encode assumptions, not laws.** Re-examine each
   restriction whenever you cross a major design boundary.
7. **Things you remove are not free.** Always check whether removing
   a cost (LDS write, AGPR shuffle, barrier) is being paid for
   elsewhere (cross-lane permute, VGPR pressure, occupancy drop).

---
Attention levers: [matrix-instructions](../02-levers/matrix-instructions.md), [memory-hierarchy](../02-levers/memory-hierarchy.md), [pipelining-scheduling](../02-levers/pipelining-scheduling.md). Attention-2D micro-levers: [knob-catalog-and-sweep](../03-autotuning/knob-catalog-and-sweep.md) (S12.1.K). Bug signatures: [failure-modes](../04-failure-reporting/failure-modes.md).
