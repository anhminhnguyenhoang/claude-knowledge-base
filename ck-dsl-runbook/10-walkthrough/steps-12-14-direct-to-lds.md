---
name: steps-12-14-direct-to-lds
description: Steps 12-14: wiring DirectToLDS (async_buffer_load_lds_addr) into gemm_universal; why DTLA alone regresses 8% but unlocks tile_k=1024 (which collapsed at 58us without it), reaching 1.01x rocBLAS with CACHE_ALL/CACHE_ALL.
source: ck-dsl-gemm-skinny-decode-README.md (lines 427-516)
---

### Step 12 — DirectToLDS (the trickiest lever) (`12_direct_to_lds.py`)

The Tensile `DTLA1_DTLB1` token maps to the hardware
`buffer_load_dwordx4 ... offen offset:0 lds` instruction — the dword
payload writes straight into LDS, bypassing the VGPR stage. Our DSL
kernel was emitting the round-trip (`global_load_dwordx4 → VGPR →
ds_write_b128`), 32 extra instructions and 32 extra VGPRs/iter.

Investigation: `ck_dsl.core.ir.IRBuilder.async_buffer_load_lds_addr` is
the DSL primitive that lowers to exactly this instruction, used by
`attention_tiled_2d.py`. `gemm_universal.py` didn't wire it.

Patch added: `TraitSpec.direct_to_lds: bool` (off by default). When
enabled, `emit_load_phase` issues `async_buffer_load_lds_addr` for both
A and B tiles, sized at `dwords=4` (16 bytes/lane). Per-pass LDS
destination advances by `block_size * 16`. The existing `b.sync()`
after the load phase already lowers to `s_waitcnt vmcnt(0) lgkmcnt(0)
; s_barrier`, so it drains in-flight DTLA writes for free.

Disassembly verifies the new path:

```text
default emit_load_phase:                with direct_to_lds=True:
  32 × global_load_dwordx4               32 × buffer_load_dwordx4 ... nt lds
  32 × ds_write_b128                     —      (eliminated!)
  32 × ds_read_b128                      32 × ds_read_b128
  16 × v_mfma_f32_16x16x32_bf16          16 × v_mfma_f32_16x16x32_bf16
```

Bit-exact correctness (max|out-ref| = 0). But the benchmark surprised:

| | best µs | %HBM | vs rocBLAS |
|---|--:|--:|--:|
| default (round-trip) | 13.30 | 31.6 % | 1.28× |
| **direct_to_lds=True** | **14.36** | 29.2 % | 1.38× |

**8 % slower.** Why? On this geometry (block_size=64 = 1 wave/CTA,
M=2, K=4096) the kernel is already wave-scheduler-bound, not VGPR /
issue-bandwidth bound. The 32 ds_writes the round-trip path emits run
in the shadow of the global_load's vmcnt, and the MFMA pipeline soaks
up whatever VGPR pressure they create. Cutting them with DTLA only
helps when the issue slots they freed can be filled with *useful work
overlapping the in-flight load* — which is exactly what Tensile's
`PGR2 PLR1 SIA3` adds (two prefetched global reads, one prefetched LDS
read, ScheduleIterAlg=3 for explicit interleaving). Our kernel waits
on `vmcnt=0` immediately, surrendering the latency-hiding DTLA was
supposed to buy.

So the rocprof name tells the whole story: `DTLA1_DTLB1` only wins
*together with* `PGR2 PLR1 SIA3`. Adding DTLA without the prefetch /
scheduling surface is a regression. The patch is kept as a documented
opt-in (`direct_to_lds: bool` in `TraitSpec`) so future ping-pong work
can build on it — see `helpers/loads.py:AsyncPingPongLoader` for the
prefetch wrapper that would compose with it.

### What still separates DSL from rocBLAS

Three Tensile tokens were needed *together* to break through:

- **DTLA1 + DTLB1** (direct-to-LDS): wired in `TraitSpec.direct_to_lds`.
- **CACHE_ALL** hint for both operands: the cache hint matters more
  than expected — `CACHE_STREAM` and non-temporal hints regress.
- **tile_k=1024**: only viable *with* DTLA. The round-trip load
  pattern collapsed at tk≥1024 (step 7: 58 µs / 7% HBM) due to
  VGPR pressure; DTLA frees those 32 VGPRs.

## Step 13 — DTLA cache-hint sweep (`13_dtl_sweep.py`)

Sweep of `tile_k ∈ {256, 512, 1024}` × `pipeline ∈ {mem, compv4}` ×
`(cache_a, cache_b) ∈ {(ALL,ALL),(ALL,STR),(STR,STR),(NT,NT),(ALL,GLC)}`.

Result: **`tk1024_mem_ALL_ALL` at 10.51 µs / 40.0% HBM — 1.01× rocBLAS.**
The shape essentially matches hipBLASLt for the first time. Without
DTLA at tk=1024, the kernel collapses to 58 µs.

Cache hints: `CACHE_ALL` wins universally for both operands. The
A tile (M=2, 16 rows of K) gets reused across CTAs (L1 reuse), so
streaming hints cost real bandwidth. B is one-shot per CTA, but
even there `STR`/`NT` shows no measurable benefit.

## Step 14 — Push tile_k past 1024 (`14_dtl_push.py`)

Does deeper K extend the win? No:
- `tk1024 mem`: 10.52 µs (1.01×)
- `tk2048 mem`: 12.02 µs (1.16×)
- `tk2048 compv4`: LDS budget exceeded (262 KiB > 160 KiB cap)

tile_k=1024 uses ~64 KiB/WG → 2 WGs/CU. tile_k=2048 forces 1 WG/CU
and the kernel becomes latency-bound on issue. The sweet spot is at
tk=1024 exactly where the LDS-vs-occupancy frontier sits.

---
Prev: [steps-06-11-existing-levers](steps-06-11-existing-levers.md). Next: [steps-15-22-multiwarp-chiplet](steps-15-22-multiwarp-chiplet.md).
