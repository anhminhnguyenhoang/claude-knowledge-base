---
name: hk-mla-decode-wiring
description: How aiter consumes HipKittens: the one experimental FP8 MLA-decode kernel, its file paths, and the gfx942/950 selection gating.
source: _sources/hipkitten-study-export.txt (lines 15-38)
---

## HipKittens in this repo

**HipKittens** is HazyResearch's C++ embedded-DSL for writing AMD GPU kernels — the AMD/HIP sibling of their ThunderKittens project for NVIDIA. It targets CDNA3/CDNA4 (MI300X, MI325X, MI350X, MI355X) and provides four primitive groups: tile types sized for matrix cores with bank-conflict-free layouts, Python-style bulk compute wrappers around HIP/inline asm, async global↔shared loads/stores, and warp-scheduling patterns (8-wave ping-pong, 4-wave interleave). Upstream: `https://github.com/HazyResearch/HipKittens` (pinned to `a5e308a…` in `aiter/jit/core.py:709`).

**Wiring into aiter**
- Pulled as a 3rd-party clone via the JIT build infra (`aiter/jit/core.py:395`, `:705-710`); `HIP_KITTENS_DIR` env var overrides the default path.
- Include path is added only when the directory exists (`aiter/jit/core.py:870-873`), so HK-using kernels compile in only after the dep is fetched.

**Current consumer: one MLA decode kernel**
- C++ side lives in `csrc/kernels/mla/hk/` — three HK header utilities (`hk_mla_utils.cuh`, `hk_mla_softmax.cuh`, `hk_mla_buffer_managers.cuh`) plus three traits headers for MI3xx tiling shapes (`mi35x_v32_fwd_decode_m16x{4,8}_fp8_fp8.cuh`, `mi3xx_v32_fwd_decode_m16x8_fp8_fp8.cuh`).
- `hk_mla_utils.cuh:13` aliases `namespace hk = kittens` and pulls in `opus/opus.hpp` (the wave-scheduling helpers). The decode traits at line 35 are specialized for DeepSeek-style MLA: `kKvLoraRank=512`, RoPE dim 64, single KV head.
- Exposed to Python as `aiter.hk_mla_decode_fwd` (declared in `csrc/include/mla.h:96`).

**Where it's selected**
`aiter/mla.py:470` gates HK over the ASM `mla_decode_stage1_asm_fwd` path. It's chosen only when:
- GPU is `gfx942` or `gfx950`,
- Q and KV are both FP8,
- `page_size ∈ {1, 64}`,
- `nhead * max_seqlen_q == 128` (gfx942/950) or `== 64` (gfx950 only),
- and `is_experimental_enabled()` is on.

So in practice it's an experimental MI300X/MI350X FP8 MLA-decode fast path covering the small-Q-token regime, gated behind an env flag and falling back to the ASM kernel otherwise. Note this is the dense MLA decode path, not part of the sparse-MLA work.

> Note: the file paths and line numbers above reflect the `aiter` tree at the time of writing — verify against the current source before relying on them.

---
Related: [overview-thesis](../10-paper/overview-thesis.md) · [pid-preprocessing-vs-algo1](pid-preprocessing-vs-algo1.md)
