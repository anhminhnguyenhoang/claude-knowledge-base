# HipKittens

Notes on **HipKittens (HK)** — HazyResearch's C++ tile-DSL for AMD CDNA3/CDNA4 GPUs (MI300X/MI325X/MI350X/MI355X), the AMD sibling of ThunderKittens — and how it shows up in the `aiter` kernel library.

All content is derived from public sources: the [arXiv paper 2511.08083](https://arxiv.org/abs/2511.08083), HazyResearch's [blog post](https://hazyresearch.stanford.edu/blog/2025-11-09-hk), the [HK GitHub repo](https://github.com/HazyResearch/HipKittens), and walkthroughs grounded in those. The verbatim Q&A export these notes were sliced from is kept in [`../_sources/`](../_sources/) for provenance.

## Reading order

Start at the bottom of the stack (hardware) and work up — every HK design choice is a response to something in `00-fundamentals`.

### 00 · Fundamentals (the hardware)
- [cu-simd-wave-vector-alu](00-fundamentals/cu-simd-wave-vector-alu.md) — what a vector ALU is and why CDNA's SIMD is 16 lanes wide; CU/SIMD/wave/quad-cycle.
- [threadblocks-and-registers](00-fundamentals/threadblocks-and-registers.md) — threadblock=workgroup terms, and the register-budget derivation of the 256×256 tile.

### 01 · The paper
- [overview-thesis](01-paper/overview-thesis.md) — the portability thesis, the 5 CDNA hardware facts, and a map of every design response + headline numbers. **Read this first.**
- [schedules](01-paper/schedules.md) — 8-wave ping-pong & 4-wave interleave, why NVIDIA wave-specialization dies on AMD, decision rubric.
- [hipcc-agpr-pinning](01-paper/hipcc-agpr-pinning.md) — the HIPCC toolchain and the AGPR-as-MFMA-input limit (~19%), bypassed via `pinned_register_tile`.
- [chiplet-scheduling](01-paper/chiplet-scheduling.md) — Algorithm 1: XCD grouping + hierarchical windowed traversal, the L2/LLC tension and L2-greedy trap.

### 02 · Applied in aiter
- [hk-mla-decode-wiring](02-aiter/hk-mla-decode-wiring.md) — the one experimental FP8 MLA-decode kernel aiter wires to HK, and its selection gating.
- [pid-preprocessing-vs-algo1](02-aiter/pid-preprocessing-vs-algo1.md) — how aiter's `pid_preprocessing.py` helpers map line-for-line onto Algorithm 1; includes an [interactive viz](assets/remap_xcd_pid_grid.html).

### Reference
- [glossary](glossary.md) — every term, deduplicated and grouped by area.

## One-sentence thesis

> Tile-based abstractions generalize across GPU vendors; the *algorithms* that instantiate them do not. HK keeps ThunderKittens' nouns (tiles, bulk ops, async load/store) but throws out wave-specialization — because AMD's static per-SIMD register partition forbids it — replacing it with 8-wave ping-pong and 4-wave interleave.
