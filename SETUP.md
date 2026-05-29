# Wiring this KB into Claude Code

This guide makes the knowledge base **permanently referenceable** in every Claude
Code session on a machine — so Claude knows the KB exists, consults it before
researching from scratch, and can extend it with the `kb-add-topic` skill. Run
these steps once per machine (e.g. after cloning the repo somewhere new).

Assumes the repo lives at `~/knowledge-base`. If you clone it elsewhere, substitute
your path everywhere below.

## 1. Make Claude aware of the KB (global CLAUDE.md)

`~/.claude/CLAUDE.md` is loaded into the system context of **every** Claude Code
session on the machine. Add a Knowledge Base section so Claude always knows the KB
is there and when to use it. Create the file if it doesn't exist, or append the
section if it does:

```markdown
## Knowledge Base
- A personal knowledge base of studied topics lives at `~/knowledge-base/` (git repo). Its root `README.md` indexes all topics; each `<topic>/README.md` gives reading order. Current topics: `hipkitten/` (AMD CDNA3/4 tile-DSL: schedules, swizzling, chiplet scheduling, aiter wiring).
- Consult it before researching a topic from scratch — if the KB already covers it, cite/build on those notes instead of re-deriving.
- To capture new learnings into it, use the `/kb-add-topic` skill (preserves source prose verbatim, adds frontmatter + cross-links + indexes). Raw sources are kept under `~/knowledge-base/_sources/`.
```

The `Current topics:` line is kept up to date automatically by the `kb-add-topic`
skill each time a topic is added.

## 2. Install the `kb-add-topic` skill machine-wide

The skill ships inside this repo at `.claude/skills/kb-add-topic/`, so it's already
available as `/kb-add-topic` whenever you work *inside this repo*. To make it
available in **any** project on the machine, copy it to the user-level skills dir:

```bash
mkdir -p ~/.claude/skills
cp -r ~/knowledge-base/.claude/skills/kb-add-topic ~/.claude/skills/
```

Verify it's picked up: in a new session the skill appears in the available-skills
list and can be invoked with `/kb-add-topic`.

### Keeping the two copies in sync

There are now two copies — the repo copy (travels with clones) and the user-level
copy (available everywhere). They're independent. After editing either, re-copy:

```bash
cp ~/knowledge-base/.claude/skills/kb-add-topic/SKILL.md ~/.claude/skills/kb-add-topic/SKILL.md
```

Alternatively, symlink the user copy at the repo copy so there's a single source of
truth (edits to the repo propagate automatically):

```bash
rm -rf ~/.claude/skills/kb-add-topic
ln -s ~/knowledge-base/.claude/skills/kb-add-topic ~/.claude/skills/kb-add-topic
```

## 3. Verify

- Open a Claude Code session in any directory and confirm `/kb-add-topic` is listed.
- Ask Claude "what's in my knowledge base?" — it should reference `~/knowledge-base`
  and the current topics without being pointed there.

## Quick reference

| Mechanism | Location | Scope | Purpose |
|---|---|---|---|
| KB content | `~/knowledge-base/` | the repo | the notes, version-controlled |
| Awareness | `~/.claude/CLAUDE.md` → Knowledge Base section | every session | Claude knows the KB exists + when to use it |
| Skill (machine-wide) | `~/.claude/skills/kb-add-topic/` | every session | one-command topic capture anywhere |
| Skill (repo) | `~/knowledge-base/.claude/skills/kb-add-topic/` | this repo + clones | skill travels with the repo |
