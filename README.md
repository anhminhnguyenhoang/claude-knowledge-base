# Knowledge Base

A personal knowledge base of things I learn — GPU kernels, hardware architecture, and adjacent systems topics. Each domain lives in its own top-level folder; raw source material is kept under [`_sources/`](_sources/) for provenance.

## Topics

| Topic | What it covers |
|---|---|
| [hipkitten/](hipkitten/) | HipKittens — HazyResearch's AMD CDNA3/4 tile-DSL: schedules, swizzling, chiplet-aware scheduling, and how `aiter` uses it. |
| [ck-dsl-runbook/](ck-dsl-runbook/) | The canonical CK DSL Optimization Runbook: the full lever catalog and method (§1–21 + decision tree): diagnose → lever families → autotune → failure modes → gfx950/CDNA4 reference. Includes the skinny-M decode GEMM worked walkthrough (4.38× slower → rocBLAS parity on MI355X) as a case study. |
| [flash-attention-v4/](flash-attention-v4/) | FlashAttention-4 — Tri Dao's Blackwell (B200) attention kernel: asymmetric hardware scaling, software-emulated exp, conditional softmax rescaling, the 5-stage warp-specialized pipeline, TMEM & 2-CTA MMA. |

## Setup

To wire this KB into Claude Code so it's referenceable in every session (and to install the `kb-add-topic` skill machine-wide), follow [`SETUP.md`](SETUP.md). Run it once per machine or after cloning somewhere new.

## Adding a topic

This repo ships a Claude Code skill, [`kb-add-topic`](.claude/skills/kb-add-topic/SKILL.md), that turns raw study material (chat exports, transcripts, paper notes) into a structured, cross-linked topic folder following the conventions below. Invoke it with `/kb-add-topic` (or just ask Claude to "add a topic to the KB") while working in this repo.

## Conventions

- **Numeric prefixes** on folders/files indicate reading order (`00-` fundamentals first).
- Each note has YAML frontmatter (`name`, `description`, `source`) and a footer linking related notes.
- Notes derived from a longer transcript cite the exact source file and line range; the transcript is preserved verbatim in `_sources/`.
- Factual claims about external papers/models carry a `Sources:` footer with primary-source links.

## License / provenance

Content is derived from public sources (papers, public repos, vendor docs) and personal study notes. See each topic's `README.md` for its specific source list.
