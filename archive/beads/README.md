# Archived task tracker

Beads was retired from this repository on 2026-10-02. Ongoing tasks are maintained
in [TASKS.md](../../TASKS.md). These files are inert historical data, not an
active tracker configuration. Do not update this snapshot when completing tasks.

## Preserved data

| File | Original path | Contents |
| --- | --- | --- |
| `issues.jsonl` | `.beads/issues.jsonl` | All 27 records, byte for byte: 10 open and 17 closed, with descriptions, metadata, close reasons, and dependency links. |
| `interactions.jsonl` | `.beads/interactions.jsonl` | Original empty interaction log (zero bytes). |
| `last-touched` | `.beads/last-touched` | Original local, ignored marker identifying `dsge-850`; retained as data, with no runtime effect. |
| `SHA256SUMS` | New integrity manifest | SHA-256 checksums for the three preserved files. |

The tracked files came from `eph/dsge` at commit
`0cfeca45eafdaefb420fb78ddba5fb1c6f319fe9`. The ignored `last-touched` marker was
copied from the main checkout on the migration date. The main checkout matched
the committed tracker snapshot. No local SQLite database, daemon state, redirects,
or additional untracked tracker data were present. All open task IDs found on
other local branches were already present in this snapshot.

The 10 migrated open IDs are `dsge-0qi`, `dsge-5fc`, `dsge-845`, `dsge-846`,
`dsge-847`, `dsge-848`, `dsge-849`, `dsge-850`, `dsge-gwm`, and `dsge-y4d`.
No issue was closed or reclassified during migration. The legacy descriptions
retain their original reproduction paths and examples as historical evidence.

## Follow-up audit

The [2026-10-02 audit](audit-2026-10-02.md) independently checks the 10 open tasks
against GitHub master and the separate security branch. All remain open; the DGS
test is partially addressed on the security branch. The dated audit script and
JSON results are new evidence, separate from the original immutable tracker data.

## Recovery and verification

Read the JSONL with any text editor or JSON parser; no Beads installation is
required. From this directory, verify the preserved bytes with:

```sh
shasum -a 256 -c SHA256SUMS
```

The removed configuration, README, ignore rules, and Git attributes remain in
Git history. For example, from the repository root:

```sh
git show 0cfeca45eafdaefb420fb78ddba5fb1c6f319fe9:.beads/config.yaml
```

To recover all previously tracked tracker files for inspection without activating
them in the working tree, write a separate archive:

```sh
git archive --format=tar 0cfeca45eafdaefb420fb78ddba5fb1c6f319fe9 .beads .gitattributes > /tmp/dsge-tracker-before-removal.tar
```

The ignored `last-touched` file is recoverable from this directory, not the old
commit. The original history and other branches were not rewritten or removed.

## Integration removed

The migration removed the active `.beads/` directory, its dedicated Git merge
attribute, and the Beads-generated session workflow in `AGENTS.md` (introduced by
commit `e084237f`, `chore: add beads release epics`). The workflow now points to the
plain-text task list. No repository hooks, Git configuration entries, or package
or CI dependencies required removal. The global Beads installation and other
projects were outside this migration's scope.
