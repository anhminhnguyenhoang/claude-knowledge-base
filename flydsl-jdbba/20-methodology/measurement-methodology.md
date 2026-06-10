---
name: measurement-methodology
description: "Measurement discipline for benchmarking a FlyDSL kernel vs an autotuning baseline (Triton): the autotune-median trap (use p10/min never median), cold-L2 do_bench vs hot-L2 rocprofv3, why CUDA-event wall-clock lies at small shapes, device-time-only via rocprofv3, and correctness via cosine+mean-error"
source: distilled from jdbba benchmarking on MI355X / gfx950 (2026-06-08..10)
---

# Measurement methodology (read before trusting any jdbba number)

Getting the *measurement* right was as load-bearing as any kernel change here. A
false "FlyDSL beats Triton 1.13–1.31×" claim survived for days purely as a
measurement artifact. These lessons generalize to any FlyDSL-vs-autotuning-baseline
comparison.

## 1. The autotune-median trap — use p10/min, NEVER median

Triton **autotunes**: a single benchmarked call fires hundreds–thousands of
trial-config dispatches, whose slow trials inflate the *median* by 30–40% (e.g.
B1024_D512: 849 dispatches spread 6307–18792 µs, median 8789). FlyDSL does not
autotune, so its median is clean. **Comparing FlyDSL's clean median to Triton's
autotune-polluted median manufactured the fake 1.3× win.**

The fair number for an autotuning kernel is its **min** (best config = what
actually deploys) or **p10** (steady-state best). `read_us2.py` defaults to p10;
for Triton you must read **min**. Kernel-name substrings on this stack:
Triton = `jagged_dense_bmm_broadcast_add`, FlyDSL = `jdbba`. (An empty/over-broad
substring silently matched the wrong kernel and reported a bogus 32.6 µs once —
match the *full* distinctive name.)

## 2. Cold-L2 (do_bench) vs hot-L2 (rocprofv3) — not noise, it changes the verdict

- `triton.testing.do_bench` **flushes the L2 each rep** (zeroes a cache-sized
  buffer) and lets the autotuner settle. This is the **cold-start** number, and it
  *includes* host launch overhead.
- `rocprofv3 --kernel-trace` runs **hot** (back-to-back dispatches, warm L2,
  device-time only) — the realistic repeated-call / MoE pattern where `Dense[b]`
  stays L2-resident across invocations.

These differ **because** the binding lever (XCD remap) is L2 reuse of `Dense[b]`:
a cold flush erodes it. So B1024_D256 reads **tie (hot) vs −3% (cold)**. Neither
is "the kernel regressing" — they **bracket reality**. Quote both and say which the
deployment resembles (repeated calls with resident weights → hot).

## 3. CUDA-event wall-clock lies at small shapes

At small / toy shapes the kernel is launch-bound and CUDA-event wall-clock is
~90% fixed host launch/dispatch overhead (~70 µs) — it reads *flat* across problem
size and masked both a real C-store waterfall and a 30% tile regression. **The
only number to optimize against is per-kernel device time from rocprofv3.** Two
separate levers looked identical in wall-clock while device time moved 30%.

There is **no pure-Python substitute** on this stack: CUDA-graph capture of the
FlyDSL launch path produces an empty graph (replay yields zeros), and batched
CUDA-event timing is still dispatch-starved.

## 4. Correctness: cosine AND mean-signed-error, on the edges

cos > 0.999 vs torch eager is the gate — but **cosine alone can mask a 123%
relative error** (the unsafe BLOCK_K=256 mis-accumulation slipped past a
cosine-only check). Use cosine *and* mean-signed-error. Always include the edge
shapes: partial bottom tile (`M_b` not a tile multiple), **empty group**
(`M_b = 0`), one-long-many-short, and `max_seq_len ≫ mean`. The bf16 path is not
bit-exact vs an fp32 reference, so don't expect `allclose`.

## 5. Change-one-thing, re-verify interleaved, re-measure surprises

- Change **one lever in isolation** on a clone; the production kernel stays
  untouched. Verify cos=1.0, *then* measure.
- GPU-clock drift between separate runs is real — re-verify the winners in a
  **single interleaved sweep** before promotion.
- **Re-measure a surprising negative on the *current* kernel** before trusting it.
  A pre-fix-clone "the XCD remap regresses under skew" finding reversed once
  measured on the bounded-A-fixed kernel — the old result was a *different bug*
  perturbing timing, not the lever.

## Harnesses (this stack)

| Harness | What it measures |
|---|---|
| `bench_headline_worker.py` + `read_us2.py` | rocprofv3 device-time, one shape, hot-L2 (p10 / min) |
| `bench_jagged_dense_bmm_perf.py` | do_bench cold-L2, all 4 headline cells, FlyDSL vs Triton |
| `bench_jdbba_vs_triton.py` | do_bench uniform + skew, through the production dispatch (end-to-end) |
| `bench_persist_xcd.py` | skew sweep of the XCD-aware visiting order vs baseline persist vs Triton |

---
The kernel these measure: [problem-and-roofline](../10-optimization-case-study/01-problem-and-roofline.md). The levers they validated: [winning-levers](../10-optimization-case-study/02-winning-levers.md).
