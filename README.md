# Knowledge Base

A personal knowledge base of things I learn — GPU kernels, hardware architecture, and adjacent systems topics. Each domain lives in its own top-level folder; raw source material is kept under [`_sources/`](_sources/) for provenance.

## Topics

| Topic | What it covers |
|---|---|
| [hipkitten/](hipkitten/) | HipKittens — HazyResearch's AMD CDNA3/4 tile-DSL: schedules, swizzling, chiplet-aware scheduling, and how `aiter` uses it. |

## Conventions

- **Numeric prefixes** on folders/files indicate reading order (`00-` fundamentals first).
- Each note has YAML frontmatter (`name`, `description`, `source`) and a footer linking related notes.
- Notes derived from a longer transcript cite the exact source file and line range; the transcript is preserved verbatim in `_sources/`.
- Factual claims about external papers/models carry a `Sources:` footer with primary-source links.

## License / provenance

Content is derived from public sources (papers, public repos, vendor docs) and personal study notes. See each topic's `README.md` for its specific source list.
