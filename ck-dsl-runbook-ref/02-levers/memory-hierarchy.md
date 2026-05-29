---
name: memory-hierarchy
description: Runbook S6: global loads/stores (coalescing, buffer-rsrc OOB), LDS/shared (async DRAM->LDS, double-buffer, swizzle), LDS bank conflicts (XOR vs padding per-arch rule), registers, and caches.
source: ck-dsl-optimization-runbook.md (lines 805-1019)
---

## 6. Memory Hierarchy

### 6.1 Global Memory Loads

- Coalesce loads. `helpers/loads.py::CoalescedTileLoader` owns the
  sync DRAM → VGPR → LDS pattern.
- Use 16 B or larger vector loads when aligned.
  `b.buffer_load_vN_f16`, `b.global_load_vN_f16` for N ∈ {1, 2, 4}
  dwords.
- Use buffer loads on AMD for robust bounds behavior:
  `b.buffer_rsrc(ptr, num_bytes)` constructs a buffer-resource
  descriptor with `flags=0x00027000`; OOB lanes return zero.
- Use raw pointer loads only when alias / alignment is clear.
- Avoid scalar loads where vector load is possible.
- Hoist invariant loads out of loops.
- Reuse loaded values across multiple outputs.
- Use read-only cache hints if available
  (`AUX = CACHE_ALL / CACHE_GLOBAL / CACHE_STREAM / NON_TEMPORAL`).
- Avoid replay from misalignment.
- Handle tails with masked loads or padded descriptors
  (`transforms.pad`/`pad_dynamic`).

### 6.2 Global Memory Stores

- Store contiguous vectors.
- Avoid per-element scalar stores in epilogues.
- Convert accumulator vectors to packed output vectors
  (`b.vec_trunc_f32_to_f16`).
- Use direct vector stores if already coalesced
  (`helpers/epilogues.py::DirectEpilogue`).
- Use LDS epilogue only when it meaningfully improves global
  coalescing (`helpers/epilogues.py::CShuffleEpilogue`). See Section
  9 for the trade-off.
- Avoid store barriers unless required for cross-thread staging.
- Ensure store validity checks do not introduce excessive branches.
- Check whether stores are 8 B, 16 B, or scalar in ISA via
  `probe_isa_inspect.py` (`buffer_store_dwordx2` / `_x4` vs
  `buffer_store_short`).

Vectorizing the epilogue is often the single largest optimization
for kernels that already have a good main loop. Replacing four scalar
`buffer_store` of `fp16` with one `fp16x8` `buffer_store` per lane
has been measured to roughly double the throughput on direct-conv
kernels — with no other change. Always inspect the epilogue ISA
(`probe_isa_inspect.py`) before tuning the main loop; the `vmem_store`
bucket reveals scalar-store kernels immediately.

### 6.3 LDS / Shared Memory

- Use LDS to share input or weights across waves / threads.
- Avoid LDS if data is not reused enough.
- Use double buffering when global load latency matters
  (`UniversalGemmSpec.pipeline = "compv4"`).
- Use direct global-to-LDS async copies when available.
  `helpers/loads.py::AsyncTileLoader` wraps the
  `raw_ptr_buffer_load_lds_addr` intrinsic with per-wave LDS base.
- Avoid register-intermediate staging if direct DMA is possible.
- Swizzle LDS layout to reduce bank conflicts
  (`helpers/layouts.py::LdsLayout`, `lds_k_pad`).
- Add padding to avoid bank conflicts.
- Match LDS write layout to LDS read layout.
- Keep LDS footprint under occupancy thresholds (verify with
  `probe_occupancy.py`).
- Reuse LDS regions after phases when lifetimes do not overlap.
- Use one shared pool for multiple phases when safe.

Switching from `buffer_load → register → vector.store LDS` to direct
DRAM-to-LDS via the AMD `raw_ptr_buffer_load_lds` /
`amd_async_buffer_load` intrinsic is worth measuring even when
register staging looks fine. A direct-conv kernel was measured to
roughly double throughput from this change alone, before any layout
modification.

Async DRAM-to-LDS instructions on AMD generally write lane-contiguous
LDS addresses (lane `i` writes at `lds_ptr + i * size`). Arbitrary
per-lane swizzled destination pointers may compile but produce wrong
output. We tested this directly: passing a per-lane swizzled LDS
pointer to `raw_ptr_buffer_load_lds` corrupted the result, so swizzle
has to be expressed in the address arithmetic of a lane-contiguous
distribution, the way CK's tile descriptors do, not by handing the
intrinsic an arbitrary LDS pointer.

`AsyncTileLoader.choose_dwords` selects `{4, 3, 1}` only (the AMDGPU
`raw_ptr_buffer_load_lds` intrinsic on this target does not accept 2
dwords).

Staging weights through LDS is not always a win. CK does it because
its tile distribution and weight read pattern fully amortize the
cost. A naive port to an MFMA kernel that already keeps weights in
registers has been measured to *regress* throughput. Test the LDS
weight path with the actual read distribution before adopting it.

Diagnostic signature: a kernel that allocates LDS for an intermediate
tensor (P in attention, A or B in GEMM) which it immediately reads
back will show `ds_write_b<N> / outer-iter ≈ tile_bytes / threads` plus
an inner-loop `s_barrier`. If the reference implementation keeps the
same tensor in registers, that LDS allocation is structural overhead.
See **§17.4** for the P-in-LDS round-trip case: 16 KiB of LDS for the
P tile plus 64 × `ds_write_b16` per K-iter, removed by switching to
register-PV + transposed-PV reads.

### 6.4 LDS Bank Conflicts

- Inspect LDS access patterns, not just total LDS bytes. Use the
  `analyze_lds_conflicts.py` tool under
  `utilities/tools/stage4_analyze/` to combine rocprof counters and
  ISA inspection.
- For AMD wave64, bank conflicts can appear with regular `(row,
  channel)` layouts.
- Try XOR swizzle on two-dimensional layouts
  (`helpers/layouts.py::LdsLayout(swizzle="xor")`).
- Try cyclic-shift swizzle when CK uses it.
- Try padding strides (`lds_k_pad` defaults `+8` sync, `0` async).
- Use transposed LDS loads where supported
  (`helpers/layouts.py::TransposeLdsReader`, `b.ds_read_tr16_b64`,
  `b.ds_read_tr_b8`).
- Count `ds_read` and `ds_write` instructions with
  `probe_isa_inspect.py`.
- Compare swizzled vs contiguous variants with
  `probe_intrinsic_counts.py`.

Cyclic-shift or XOR swizzle on a 2D tile layout only pays off in our
experiments when paired with the matching async load distribution.
With register-staged LDS writes, a manually-swizzled LDS layout has
been measured to regress slightly. Bank-conflict reduction alone is
not enough; the swizzle has to be aligned with how the load is
actually delivered to LDS.

#### 6.4a LDS Swizzle Strategy Selection: XOR vs Padding

Notation:

- **LDS_total**: Total LDS allocated by your kernel (per CU). From
  `probe_occupancy.py` or `analyze_hsaco`.
- **LDS_avail**: Available LDS per CU on the target architecture.

| Architecture | GPU | LDS per CU | Preferred swizzle |
|---|---|---|---|
| gfx90a | MI210 / MI250X | 64 KB | XOR (capacity precious) |
| gfx942 | MI300 | 64 KB | XOR (capacity precious) |
| gfx950 | MI350X / MI355X | 160 KB | padding (capacity abundant) |

Two primary approaches exist for eliminating LDS bank conflicts, with
dramatically different performance characteristics depending on GPU
architecture:

**XOR Swizzle**: Mathematically proven bank-conflict-free addressing
using bitwise XOR operations.

- Cost: higher ALU overhead (more complex address computation; 5-7
  ALU vs 1-2 ALU per LDS access).
- LDS overhead: zero — no wasted LDS bytes.
- Best for: architectures with limited LDS capacity.
- Preferred when: LDS footprint is near capacity limits and occupancy
  depends on fitting in available LDS.
- Eliminates bank conflicts completely through address computation.

**Padding Swizzle**: Simple padding to avoid conflict patterns (e.g.,
`K=64 → K_PAD=72`).

- Cost: lower ALU overhead (simple offset calculation).
- LDS overhead: small percentage of total LDS (typically negligible
  on high-capacity architectures).
- Best for: architectures with abundant LDS capacity.
- Preferred when: LDS capacity is not the bottleneck and simpler
  addressing improves throughput.
- Eliminates bank conflicts through strategic padding.

Architecture-specific selection rule:

```python
if LDS_total < 96 * 1024:
    use_xor_swizzle()      # capacity-constrained
elif LDS_avail >= 128 * 1024:
    use_padding_swizzle()  # abundant LDS
else:
    benchmark_both()       # 96 KB <= LDS_b < 128 KB
```

**gfx950 caveat**: Explicit scheduling barriers (`s_sched_barrier`,
`s_sched_group_barrier`) are silently removed by the LLVM backend on
gfx950. Verify with `probe_isa_inspect.py` (the `sched_barrier`
sub-bucket should be 0). Cannot rely on scheduling barriers to
mitigate XOR overhead on gfx950.

Critical asymmetry: `ds_write_b128` has 32-dword conflict period
while `ds_read_b128` has 64-dword period. Read and write conflict
behavior can differ — may need separate swizzle strategies. See
`utilities/skills/empirical-case-studies.md` (Case Study 2) and
**§21.2** for the full per-arch LDS table.

### 6.5 Registers

- Track VGPR and SGPR usage with `probe_occupancy.py`.
- High VGPR can reduce occupancy.
- Too-low VGPR cap can spill or reduce scheduling freedom.
- Hoist invariants but avoid keeping large temporary structures
  alive.
- Store only scalar state for tile windows / descriptors.
- Avoid persistent heavyweight objects in kernels.
- Prefer precomputed offsets over live descriptor state.
- Use compile-time constants to reduce scalar arithmetic
  (`IRBuilder.const_i32`).
- Watch register growth when adding double buffering or extra
  accumulators.
- Set `kernel.attrs["waves_per_eu"]` to override LLVM's
  occupancy hint when the compiler is wrong.

### 6.6 Caches

- Determine whether the benchmark uses warm cache or cold cache.
- Use rotating buffers if measuring bandwidth realistically.
- Use graph replay to separate cache behavior from launch overhead.
- Do not tune only for warm cache unless production has warm cache.
- Consider L2 prefetch only when the memory pattern is predictable.

---
Epilogue store path: [epilogue](epilogue.md). Async copy + pipeline: [pipelining-scheduling](pipelining-scheduling.md). LDS swizzle knobs: [knob-catalog-and-sweep](../03-autotuning/knob-catalog-and-sweep.md) (S12.1.F); per-arch LDS table: [target-architecture-gfx950](../06-reference/target-architecture-gfx950.md).
