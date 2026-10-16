import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

WORKFLOW: Final = Path(__file__).parents[1] / "workflows" / "carto-upstream-sync-ready-checker.yml"
REQUIRED: Final = tuple(
    yaml.safe_load((Path(__file__).parents[1] / "carto-sync-required-workflows.yml").read_text())["required"]
)

FAKE_GH: Final = """#!/usr/bin/env bash
case "$1 $2" in
  "pr list") printf '%s' "$FAKE_PR_JSON" ;;
  "pr checks") printf '%s' "$FAKE_CHECKS" ;;
  "pr edit") echo "$*" >> "$EDIT_LOG" ;;
  *) exit 9 ;;
esac
"""
FAKE_YQ: Final = "#!/usr/bin/env bash\nprintf '%s' \"$FAKE_REQUIRED\"\n"


def checker_script() -> str:
    steps: Final = yaml.safe_load(WORKFLOW.read_text())["jobs"]["check-and-label"]["steps"]
    script: Final = next(step["run"] for step in steps if step.get("id") == "check-and-label")
    return script.replace("${{ github.event.workflow_run.head_branch }}", "upstream-sync/v9.9.9").replace(
        "${{ github.repository }}", "CartoDB/litellm"
    )


def checks(**buckets: str) -> list[dict[str, str]]:
    return [{"name": wf, "workflow": wf, "bucket": buckets.get(wf, "pass")} for wf in REQUIRED]


def label_edit(tmp_path: Path, labeled: bool, mergeable: str, check_list: list[dict[str, str]]) -> str | None:
    for tool, body in (("gh", FAKE_GH), ("yq", FAKE_YQ)):
        path = tmp_path / tool
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    labels: Final = [{"name": "upstream-sync"}, *([{"name": "sync-ready"}] if labeled else [])]
    env: Final = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
        "EDIT_LOG": str(tmp_path / "edits"),
        "FAKE_PR_JSON": json.dumps({"number": 1, "mergeable": mergeable, "labels": labels, "url": "u", "title": "t"}),
        "FAKE_CHECKS": json.dumps(check_list),
        "FAKE_REQUIRED": json.dumps(REQUIRED),
    }
    subprocess.run(["bash", "-c", checker_script()], env=env, check=True, capture_output=True)
    edits: Final = (tmp_path / "edits").read_text() if (tmp_path / "edits").exists() else ""
    if "--remove-label sync-ready" in edits:
        return "remove"
    return "add" if "--add-label sync-ready" in edits else None


@pytest.mark.parametrize(
    ("labeled", "mergeable", "check_list", "expected"),
    [
        (True, "MERGEABLE", checks(**{"CARTO Features Manifest Check": "fail"}), "remove"),
        (True, "MERGEABLE", checks(**{"CARTO Feature Tests": "pending"}), "remove"),
        (True, "MERGEABLE", checks()[:2], "remove"),
        (True, "CONFLICTING", checks(), "remove"),
        (True, "UNKNOWN", checks(**{"CARTO Feature Tests": "fail"}), None),
        (True, "MERGEABLE", checks(), None),
        (False, "MERGEABLE", checks(), "add"),
        (False, "MERGEABLE", checks(**{"CARTO Feature Tests": "fail"}), None),
    ],
    ids=[
        "stale-label-removed-on-failure",
        "stale-label-removed-while-pending",
        "stale-label-removed-when-required-missing",
        "stale-label-removed-on-conflict",
        "unknown-mergeability-waits",
        "passing-head-keeps-label",
        "passing-head-gets-label",
        "failing-head-stays-unlabeled",
    ],
)
def test_sync_ready_label_tracks_the_current_head(
    tmp_path: Path, labeled: bool, mergeable: str, check_list: list[dict[str, str]], expected: str | None
) -> None:
    assert label_edit(tmp_path, labeled, mergeable, check_list) == expected


def test_runs_on_failed_completions_too() -> None:
    job: Final = yaml.safe_load(WORKFLOW.read_text())["jobs"]["check-and-label"]
    assert "if" not in job
