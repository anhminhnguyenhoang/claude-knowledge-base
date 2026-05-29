---
name: kb-add-topic
description: >-
  Turn raw study material (chat exports, transcripts, paper notes, polished
  markdown) on a topic into a structured, cross-linked knowledge-base topic
  folder inside this repo. Use when the user wants to "add a topic to the KB",
  "build a knowledge base for X", "organize these notes into the KB", or
  capture what they've learned about something into durable, navigable notes.
---

# kb-add-topic

Build one KB topic folder from raw source material, preserving the user's
validated prose **verbatim** while adding navigation (frontmatter, indexes,
cross-links, a merged glossary). This skill encodes the workflow used to create
the `hipkitten/` topic — read that folder as the reference example.

## Core principle

**Preserve, don't rewrite.** The source material is often already accurate and
reviewed. Rewriting risks introducing errors into validated content. Default to
slicing the source by topic and wrapping each slice, not paraphrasing it. Only
rewrite when the user explicitly asks.

## Repo conventions (match these exactly)

```
<repo-root>/
├── README.md                  # top index: table of topics + conventions
├── _sources/                  # raw source files, verbatim, for provenance
└── <topic>/
    ├── README.md              # topic index + reading order + 1-sentence thesis
    ├── glossary.md            # merged, deduped glossary (if sources have them)
    ├── 00-<area>/             # numeric prefixes = reading order; low = foundations
    ├── 10-<area>/
    ├── 20-<area>/
    └── assets/                # diagrams, interactive HTML, images
```

- **Numeric folder prefixes** (`00-`, `10-`, `20-`) encode reading order, lowest
  = read first (usually fundamentals).
- **Every note** starts with YAML frontmatter: `name` (kebab-case slug matching
  the filename), `description` (one specific line), and `source` (the file + line
  range, or "verbatim" for whole-file imports).
- **Every note** ends with a footer of `[label](relative/path.md)` cross-links to
  related notes.
- Notes sliced from a transcript cite the exact `source: _sources/<file> (lines A-B)`.
- Factual claims about external papers/models keep their original `Sources:` footer
  with primary-source links. Never strip citations.

## Workflow

### 1. Gather & triage sources
- Identify every input file the user points at (and ask if scope is unclear).
- Read them fully. Classify each: **primary** (the canonical content), **redundant**
  (e.g. a raw terminal capture of a session that's also cleanly exported — drop it),
  **polished** (already-good standalone markdown — import whole), **asset** (HTML/img),
  **out-of-scope** (exclude, tell the user why).
- For a long transcript, map exact section boundaries with `grep -nE` on headers
  (`^#`, `^##`) and any transcript markers (`^result:`, `^\[USER\]`, timestamps) so
  you can slice precise line ranges.

### 2. Propose structure, then confirm
- Draft the folder tree and the note list. Map each planned note to a source line
  range or file.
- Use `AskUserQuestion` to confirm the decisions that actually branch the work:
  - **Content treatment**: preserve-verbatim (default) vs refactor-into-atomic vs hybrid.
  - **Scope**: topic-only repo vs general KB with this as one topic (this repo is the latter).
  - **Build location / push**: build locally vs in-repo; who runs the GitHub push.
- Don't ask about things you can infer or default sensibly.

### 3. Build the skeleton
- Create `<topic>/` with its area subfolders and `assets/`; ensure `_sources/` exists.
- Copy raw source files into `_sources/` and assets into `<topic>/assets/` verbatim.

### 4. Slice notes verbatim
- For transcript-derived notes, emit each as: frontmatter + `sed -n 'A,Bp'` slice +
  cross-link footer. A small shell helper keeps this consistent (see the example in
  step 6 of the hipkitten build; reproduced below).
- For polished standalone markdown, prepend frontmatter and append a footer; fix any
  intra-doc relative links (e.g. a referenced asset now living under `assets/`).

```bash
note() {  # $1=outfile $2=name $3=desc $4=start $5=end $6=footer  (SRC = source path)
  { printf -- '---\nname: %s\ndescription: %s\nsource: %s (lines %s-%s)\n---\n\n' \
        "$2" "$3" "$(basename "$SRC")" "$4" "$5"
    sed -n "${4},${5}p" "$SRC"
    printf '\n\n---\n%s\n' "$6"
  } > "$1"
}
```

### 5. Merge the glossary
- If multiple source sections each carry their own glossary, merge into one
  `<topic>/glossary.md`, deduped and grouped by area (hardware / instructions /
  concepts / etc.). Keep a `Sources:` footer.

### 6. Write the indexes
- `<topic>/README.md`: what the topic is, all sources it derives from, a reading
  order (linked list grouped by area folder), and a one-sentence thesis.
- Root `README.md`: add/refresh the topics table with a row for this topic. Restate
  conventions once here.

### 7. Verify (do not skip)
- Sweep for transcript bleed: `grep -rn '^result:' <topic>/` and any timestamp/
  speaker markers — these mean a slice ran one line too long. Trim them.
- Check each note's first body line after frontmatter is a real heading/sentence,
  and each tail ends cleanly (no dangling `result:` summary).
- Confirm every cross-link path resolves (relative to the note's location).
- Add a "verify paths against current source" caveat to any note that cites
  live code paths / line numbers, since those drift.

### 8. Commit (and push only if asked)
- If this is a new repo: `git init`, add a `.gitignore`, stage, first commit.
- Otherwise: stage the new topic + updated root README, one focused commit.
- **Identity**: this is a personal repo — before committing, confirm the commit
  email (offer GitHub no-reply vs a personal address vs the global default) unless
  the user has already stated a preference. Set it repo-local (`git config user.email`),
  never global.
- **Push**: only when the user asks. `gh` may not be authenticated — if so, hand the
  user the exact `gh auth login` + `gh repo create ... --source . --push` commands to
  run themselves rather than failing silently.

## Anti-patterns
- Rewriting validated prose "to make it more KB-idiomatic" (introduces errors).
- Slicing line ranges loosely so transcript artifacts (`result:`, timestamps) leak in.
- Stripping a note's original source citations.
- Committing with the wrong identity, or pushing without being asked.
- Burying foundations — if a topic's design choices respond to hardware/primitives,
  those go in `00-` and the README says to read them first.
