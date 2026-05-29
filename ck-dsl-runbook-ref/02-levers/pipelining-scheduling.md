---
name: pipelining-scheduling
description: Runbook S8: software pipeline (prologue/steady/epilogue, ping-pong, early-V), async copy and wait placement, waits/barriers, scheduling hints (silently dropped on gfx950), and instruction-level balance.
source: ck-dsl-optimization-runbook.md (lines 1106-1175)
---

## 8. Pipelining And Scheduling

### 8.1 Software Pipeline

- Separate prologue, steady-state loop, and epilogue. The DSL exposes
  `helpers/pipeline.py::SoftwarePipeline.run_ping_pong(...)` for the
  ping-pong staging.
- Prefetch next tile while computing current tile.
- Use ping-pong buffers (typically two LDS halves).
- Use more stages only if latency is not hidden.
- Keep pipeline state small.
- Ensure waits occur as late as safely possible.
- Avoid barrier before useful independent compute.
- Order async issues to give *all* dependent compute a chance to
  overlap. The "early-V" schedule in **§17.4** moved the V async
  copy issue from after QK to before QK so V overlaps with both QK
  and softmax, not just softmax — see §17.4 for the cohort result.

### 8.2 Async Copy

- Prefer direct global-to-LDS instructions if available
  (`AsyncTileLoader` + `b.async_buffer_load_lds`).
- Track outstanding async operations.
- Place waits just before consumers (`b.s_waitcnt(vmcnt=0)`).
- Ensure boundary lanes write zeros or that suppressed loads do not
  leave stale LDS.
- Understand whether async instruction writes lane-contiguous or
  arbitrary addresses (lane-contiguous on AMD; see Section 6.3).
- If arbitrary swizzle is needed, verify the intrinsic supports it.

### 8.3 Waits And Barriers

- Count `s_waitcnt` and `s_barrier` with `probe_isa_inspect.py`.
- Distinguish global memory waits from LDS waits.
- Use barriers only for cross-thread visibility.
- `s_waitcnt` does NOT replace a barrier when another thread reads
  your LDS write.
- Avoid barrier after direct global store unless needed.
- Collapse multiple waits where possible.
- Avoid compiler-generated over-waiting by separating phases.
- The DSL encodes the canonical `s_waitcnt(vmcnt=16, lgkmcnt=16)`
  value as the constant `20336`; cross-check in the lowered IR.

### 8.4 Scheduling Hints

- Try compiler scheduling flags only after correctness is stable.
- On AMD, experiment with `sched_group_barrier(mask, count, sync_id)`,
  `s_setprio`, and `s_waitcnt` from `helpers/schedule.py::SchedulePolicy`.
- Compare ISA with and without hints (`probe_isa_inspect.py`,
  `probe_intrinsic_counts.py`).
- Validate correctness after every scheduler flag.
- Some flags can break generated kernels even when they work for
  hand-written HIP.

On gfx950 the LLVM backend silently removes explicit
`s_sched_barrier` / `s_sched_group_barrier` instructions; verify with
`probe_isa_inspect.py` (the `sched_barrier` sub-bucket should be 0)
or `probe_intrinsic_counts.py` (`sched.barrier` / `sched.group.barrier`
should be 0 after lowering on gfx950).

### 8.5 Instruction-Level Balance

- Interleave MFMA, LDS reads, and global loads.
- Avoid long runs of memory instructions followed by long runs of
  MFMA if latency is exposed.
- Avoid immediate use of a just-loaded value if independent work
  exists.
- Group enough MFMA between waits.
- Watch scalar ALU address arithmetic (`probe_isa_inspect.py`'s
  `salu` sub-bucket grows if the address math is not hoisted).

---
Pipeline knobs (mem/compv3/compv4): [knob-catalog-and-sweep](../03-autotuning/knob-catalog-and-sweep.md) (S12.1.D). Async DRAM->LDS contract: [memory-hierarchy](memory-hierarchy.md). gfx950 sched-barrier caveat: [target-architecture-gfx950](../06-reference/target-architecture-gfx950.md).
