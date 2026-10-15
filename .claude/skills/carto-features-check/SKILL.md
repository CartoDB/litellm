---
name: carto-features-check
description: Verify the CARTO features manifest (.github/carto-features.yml) and check that CARTO customizations in a diff have a corresponding manifest entry. Use when the user says "check carto features", "features manifest", "did I register this feature", "verify carto manifest", "carto-features.yml", before opening a PR against carto/main, or after adding a CARTO customization to a provider transformation.
---

# CARTO Features Manifest Check

CARTO maintains a fork of LiteLLM. Custom features ("CARTO customizations") that are
NOT upstream must survive periodic upstream syncs. They are protected by a tripwire:
`.github/carto-features.yml` stores a `(pattern, file)` fingerprint of each
customization, and the `carto-features-check.yml` workflow greps for those patterns
on every `upstream-sync/**` push and every PR into `carto/main`. A missing pattern
fails CI — that is the signal a sync silently dropped a customization.

This skill does two jobs:

1. **Verify** (deterministic) — run the CI grep logic locally so you catch a broken
   manifest before pushing. This mirrors the workflow exactly.
2. **Coverage** (diff-driven + judgment) — given the current branch's changes, find
   CARTO customizations that have NO manifest entry and help add them.

> Important honesty note: there is **no reliable machine marker** for "this code is a
> CARTO customization." Only a fraction of customizations carry a `# CARTO:` comment.
> So the coverage job is heuristic — it surfaces candidates from the diff and relies on
> judgment to decide what is a real customization worth protecting. Do not claim full
> coverage; claim "candidates found in this diff."

## Job 1 — Verify the manifest (deterministic)

Run this from the repo root. It is the same logic as `.github/workflows/carto-features-check.yml`,
kept as an inline copy here on purpose (this skill is self-contained; the workflow is
the source of truth for CI).

```bash
python3 - <<'PY'
import yaml, subprocess, sys
with open('.github/carto-features.yml') as f:
    data = yaml.safe_load(f)
total = 0; missing = []
for feat in data.get('features', []):
    for v in feat.get('verification', []):
        total += 1
        r = subprocess.run(['grep', '-q', v['pattern'], v['file']], capture_output=True)
        ok = r.returncode == 0
        print(f'{"OK " if ok else "MISS"} {feat["name"]} :: {v["pattern"]} in {v["file"]}')
        if not ok:
            missing.append(f'{feat["name"]}: {v["pattern"]} in {v["file"]}')
print(f'--- {total - len(missing)}/{total} verified ---')
if missing:
    print('MISSING:')
    for m in missing:
        print(f'  - {m}')
    sys.exit(1)
PY
```

- Exit 0 → manifest is consistent with the code.
- Exit 1 → a registered pattern is gone. Either the customization was lost in a sync
  (restore it) or it was intentionally refactored (update the pattern in the manifest
  to a new stable marker). Decide which with the user — do not silently rewrite a pattern
  to make CI green without confirming the feature still works.

## Job 2 — Coverage: did new CARTO customizations get registered?

The goal: find code in this branch that is a CARTO customization but is **not** covered
by any manifest entry.

### Step 1 — Get the changed files

Compare against the fork's mainline (`carto/main`) for a branch/PR, or against the
last upstream sync point if auditing the whole fork.

```bash
# files changed in this branch vs carto/main
git diff --name-only carto/main...HEAD -- 'litellm/**/*.py'
```

Focus on provider transformations and core utils — that is where customizations live:
`litellm/llms/**/transformation.py`, `litellm/responses/**`,
`litellm/litellm_core_utils/**`, `litellm/llms/**/common_utils.py`.

### Step 2 — Identify customization candidates

For each changed file, look at the added lines (`git diff carto/main...HEAD -- <file>`)
and flag a candidate when the change is CARTO-specific behavior rather than a plain
upstream sync. Signals (any of):

- A `# CARTO:` comment.
- A new helper function added by a CARTO PR (e.g. `_strip_openai_annotations`,
  `_normalize_empty_tool_call_arguments`, `_content_to_text_string`).
- Provider request/response rewriting that works around a provider quirk
  (Cortex error codes, Databricks `INVALID_PARAMETER_VALUE`, Azure URL stripping).
- Anything tied to a CARTO `source_prs` number in the PR description.

Cross-check the file against the upstream commit history if unsure:
`git log --oneline -- <file>` — author `@berri.ai` / upstream merge = upstream;
a CartoDB author on a `fix/` or `sc-*` branch = likely a customization.

### Step 3 — Check manifest coverage for each candidate

A candidate is **covered** if some manifest `verification` entry greps a pattern that
sits inside the candidate's added code. Read `.github/carto-features.yml` and match by
`file` + whether the pattern lands on a changed line.

```bash
git diff carto/main...HEAD -- litellm/llms/databricks/chat/transformation.py | grep '^+' | grep -E 'def |# CARTO'
```

If a candidate's distinctive markers do not appear in any manifest pattern → **uncovered**.

### Step 4 — Report and (with approval) register

Report uncovered candidates as a table: `feature (proposed name) | file | suggested marker(s) | source PR`.
Then, only after the user confirms which are real customizations, add entries to
`.github/carto-features.yml` using this schema:

```yaml
  - name: "Human Readable Feature Name"
    source_prs: [<PR numbers>]
    description: "What the customization does and which provider quirk it works around."
    files:
      - <path to file(s) the feature lives in>
    verification:
      - pattern: "<unique stable string in the added code>"
        file: <path>
```

Only `verification` (`pattern` + `file`) is used by CI. `name`, `source_prs`,
`description`, `files` are documentation. Picking good patterns matters:

- **Prefer stable, unique markers**: a function `def _name(`, a distinctive error
  string, or an intentional `# CARTO:` anchor comment. These survive reformatting.
- **Avoid generic tokens** (`return`, `content`, `if message:`) — they match upstream
  code too and give false greens.
- Patterns are matched with plain `grep` (BRE). Parentheses are literal, `.` matches any
  char — fine for `def foo(` style markers. Avoid relying on regex metachars.
- **Verify presence before committing**: `grep -c "<pattern>" <file>` should return ≥1.

### Step 5 — Re-run Job 1

After adding entries, run the Job 1 verify block. It must report all patterns OK
(exit 0) before opening/updating the PR.

## Notes

- This skill never edits production code. It only reads diffs and edits
  `.github/carto-features.yml`.
- If `.github/carto-features.yml` is absent, the CI check skips silently — but for this
  fork it should always exist. Treat its absence as a problem to flag, not ignore.
