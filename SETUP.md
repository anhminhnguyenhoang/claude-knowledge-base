# Wiring this KB into Claude Code

This guide makes the knowledge base **permanently referenceable** in every Claude
Code session on a machine — so Claude knows the KB exists, consults it before
researching from scratch, and can extend it with the `kb-add-topic` skill. Run
these steps once per machine (e.g. after cloning the repo somewhere new).

Assumes the repo lives at `~/knowledge-base`. If you clone it elsewhere, substitute
your path everywhere below.

> **Carrying everything to a new machine.** Steps 1–2 wire up the KB itself. To
> reproduce your *full* Claude Code working environment on a fresh box — the global
> instructions, the user-level skills, and the accumulated per-project memory — see
> [§4 Carry your global config](#4-carry-your-global-config-claudemd--skills) and
> [§5 Carry your auto-memory](#5-carry-your-auto-memory-across-machines). The
> [Quick reference](#quick-reference) table at the bottom lists every moving part
> and where it lives.

## 1. Make Claude aware of the KB (global CLAUDE.md)

`~/.claude/CLAUDE.md` is loaded into the system context of **every** Claude Code
session on the machine. Add a Knowledge Base section so Claude always knows the KB
is there and when to use it. Create the file if it doesn't exist, or append the
section if it does:

```markdown
## Knowledge Base
- A personal knowledge base of studied topics lives at `~/knowledge-base/` (git repo). Its root `README.md` indexes all topics; each `<topic>/README.md` gives reading order. Current topics: `hipkitten/`, `ck-dsl-runbook/`, `flash-attention-v4/`, `flydsl-jdbba/` (see the root README for one-line summaries of each).
- Consult it before researching a topic from scratch — if the KB already covers it, cite/build on those notes instead of re-deriving.
- To capture new learnings into it, use the `/kb-add-topic` skill (preserves source prose verbatim, adds frontmatter + cross-links + indexes). Raw sources are kept under `~/knowledge-base/_sources/`.
```

The `Current topics:` line is kept up to date automatically by the `kb-add-topic`
skill each time a topic is added. (The exact wording in your live `~/.claude/CLAUDE.md`
may carry a longer per-topic summary — that's fine; this is just the seed form.)

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

## 4. Carry your global config (CLAUDE.md + skills)

Beyond KB awareness, `~/.claude/CLAUDE.md` accumulates **global operating
instructions** that should follow you to every machine (environment rules, output
conventions, merge-conflict discipline, the Knowledge Base section, …), and
`~/.claude/skills/` holds **user-level skills** available in every project
(`kb-add-topic`, `cdna-kernel-opt`, `chiplet-xcd-remap`, `kernel-walkthrough`, …).

These live **outside any repo**, so they don't travel automatically. To snapshot
them onto a new system, copy both to a synced location (this KB repo, a dotfiles
repo, or a cloud drive) and restore on the target:

```bash
# --- backup (on the old machine) ---
DEST=~/claude-config-backup        # or a dotfiles repo, Drive folder, etc.
mkdir -p "$DEST"
cp ~/.claude/CLAUDE.md "$DEST/CLAUDE.md"
cp -r ~/.claude/skills "$DEST/skills"

# --- restore (on the new machine, after installing Claude Code) ---
mkdir -p ~/.claude
cp ~/claude-config-backup/CLAUDE.md ~/.claude/CLAUDE.md
cp -r ~/claude-config-backup/skills/. ~/.claude/skills/
```

If the KB repo is your sync vehicle, the repo already carries the `kb-add-topic`
skill under `.claude/skills/` (step 2) — but `CLAUDE.md` and the *other* user
skills are **not** in the repo by default. Either copy them in deliberately, or
keep them in a separate private dotfiles repo (recommended if `CLAUDE.md` contains
anything host-specific you don't want public).

## 5. Carry your auto-memory across machines

Claude Code's persistent memory lives **per project** under
`~/.claude/projects/<slugified-project-path>/memory/` — an indexed set of markdown
files (`MEMORY.md` + topic files) that persist what Claude has learned about you,
your feedback, and each project across sessions. It is **not** in any repo and is
**namespaced by the project's absolute path**, so it does not travel automatically
and the directory name changes if the project lives at a different path on the new
machine.

```bash
# --- backup (on the old machine): snapshot ALL projects' memory ---
DEST=~/claude-config-backup
mkdir -p "$DEST"
tar czf "$DEST/claude-memory.tgz" -C ~/.claude/projects \
  $(cd ~/.claude/projects && find . -maxdepth 2 -name memory -type d)

# --- restore (on the new machine) ---
mkdir -p ~/.claude/projects
tar xzf ~/claude-config-backup/claude-memory.tgz -C ~/.claude/projects
```

**Path-namespace caveat (important).** The slug encodes the project's absolute
path: `/home/anguyenh/FlyDSL` → `-home-anguyenh-FlyDSL`. If the project lives at a
**different** path on the new machine (different username or checkout location),
rename the restored directory to match the new slug, or Claude won't find the
memory:

```bash
# example: same layout, different username `me` instead of `anguyenh`
mv ~/.claude/projects/-home-anguyenh-FlyDSL \
   ~/.claude/projects/-home-me-FlyDSL
```

Keep the layout identical across machines (same home path + checkout dir) and no
rename is needed. Re-run this snapshot periodically — memory keeps growing as you
work, so the backup is only as fresh as your last copy.

> **Why not commit memory into this repo?** Memory is per-project, churny, and may
> reference private project context — versioning a snapshot would drift from the
> live files immediately and risk leaking that context publicly. Snapshot/restore
> via a private location (above) keeps it portable without those downsides.

## Quick reference

| Mechanism | Location | Scope | Travels how |
|---|---|---|---|
| KB content | `~/knowledge-base/` | the repo | `git clone` |
| Awareness | `~/.claude/CLAUDE.md` → Knowledge Base section | every session | §4 backup/restore |
| Global instructions | `~/.claude/CLAUDE.md` (whole file) | every session | §4 backup/restore |
| Skill (machine-wide) | `~/.claude/skills/<name>/` | every session | §4 backup/restore |
| Skill (repo, KB only) | `~/knowledge-base/.claude/skills/kb-add-topic/` | this repo + clones | `git clone` |
| Auto-memory | `~/.claude/projects/<path-slug>/memory/` | per project, every session | §5 tar snapshot (mind the path slug) |
