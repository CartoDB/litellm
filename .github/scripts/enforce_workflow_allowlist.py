#!/usr/bin/env python3
"""Disable every workflow outside .github/carto-enabled-workflows.yml and the carto-* prefix.

Upstream syncs bring new workflows that start enabled, so this runs after each sync and on a schedule
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import yaml

ALLOWLIST: Final = Path(".github/carto-enabled-workflows.yml")
WORKFLOW_DIR: Final = PurePosixPath(".github/workflows")


@dataclass(frozen=True, slots=True)
class Workflow:
    id: int
    path: str
    state: str


def is_kept(path: str, keep: frozenset[str]) -> bool:
    pure: Final = PurePosixPath(path)
    return pure.parent != WORKFLOW_DIR or pure.name.startswith("carto-") or path in keep


def workflows_to_disable(workflows: Iterable[Workflow], keep: frozenset[str]) -> tuple[Workflow, ...]:
    return tuple(sorted((w for w in workflows if w.state == "active" and not is_kept(w.path, keep)), key=lambda w: w.path))


def load_keep(allowlist: Path) -> frozenset[str]:
    return frozenset(yaml.safe_load(allowlist.read_text())["keep"])


def gh(*args: str) -> str:
    return subprocess.run(("gh", *args), check=True, capture_output=True, text=True).stdout


def list_workflows(repo: str) -> tuple[Workflow, ...]:
    out: Final = gh("api", "--paginate", f"repos/{repo}/actions/workflows", "--jq", ".workflows[] | {id, path, state}")
    return tuple(Workflow(**json.loads(line)) for line in out.splitlines())


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args: Final = parser.parse_args()
    repo: Final = os.environ["GITHUB_REPOSITORY"]

    keep: Final = load_keep(ALLOWLIST)
    missing: Final = sorted(path for path in keep if not Path(path).is_file())
    if missing:
        print(f"::error::Allowlisted workflows do not exist: {', '.join(missing)}", file=sys.stderr)
        return 1

    targets: Final = workflows_to_disable(list_workflows(repo), keep)
    for workflow in targets:
        if not args.dry_run:
            gh("workflow", "disable", str(workflow.id), "--repo", repo)
        print(f"{'would disable' if args.dry_run else 'disabled'} {workflow.path}")
    print(f"{len(targets)} workflow(s) {'to disable' if args.dry_run else 'disabled'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
