#!/usr/bin/env python3
"""Keep the upstream-sync workflow_run triggers in line with carto-sync-required-workflows.yml.

GitHub cannot load workflow_run triggers from a file, so the names are repeated. The ready checker
must trigger on every required workflow (it is only re-invoked when a listed one completes), and the
CI fixer and customizations analyzer must only react to required ones
"""

import sys
from pathlib import Path
from typing import Final, cast

import yaml

WORKFLOWS_DIR: Final = Path(".github/workflows")
REQUIRED_FILE: Final = Path(".github/carto-sync-required-workflows.yml")
READY_CHECKER: Final = "carto-upstream-sync-ready-checker.yml"
SUBSET_TRIGGERED: Final = ("carto-upstream-sync-ci-fixer.yml", "carto-upstream-sync-customizations-analyzer.yml")


def load(path: Path) -> dict[object, object]:
    return cast(dict[object, object], yaml.safe_load(path.read_text()))


def triggers(workflow_file: str) -> frozenset[str]:
    # PyYAML parses the bare `on:` key as boolean True
    on: Final = cast(dict[str, dict[str, list[str]]], load(WORKFLOWS_DIR / workflow_file)[True])
    return frozenset(on["workflow_run"]["workflows"])


def problems() -> tuple[str, ...]:
    required: Final = frozenset(cast(list[str], load(REQUIRED_FILE)["required"]))
    existing: Final = frozenset(cast(str, load(path).get("name")) for path in WORKFLOWS_DIR.glob("*.y*ml"))
    ready: Final = triggers(READY_CHECKER)
    return (
        *(f"required workflow '{name}' does not exist in {WORKFLOWS_DIR}" for name in sorted(required - existing)),
        *(f"{READY_CHECKER} does not trigger on required '{name}'" for name in sorted(required - ready)),
        *(f"{READY_CHECKER} triggers on non-required '{name}'" for name in sorted(ready - required)),
        *(
            f"{workflow} triggers on non-required '{name}'"
            for workflow in SUBSET_TRIGGERED
            for name in sorted(triggers(workflow) - required)
        ),
    )


def main() -> int:
    found: Final = problems()
    for problem in found:
        print(f"::error::{problem}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
