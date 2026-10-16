import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

WORKFLOW: Final = Path(__file__).parents[1] / "workflows" / "carto-upstream-sync-ci-fixer.yml"
REQUIRED: Final = tuple(
    yaml.safe_load((Path(__file__).parents[1] / "carto-sync-required-workflows.yml").read_text())["required"]
)
UPSTREAM: Final = "MCP OAuth happy path"

FAKE_GH: Final = """#!/usr/bin/env bash
case "$1 $2" in
  "pr list" | "pr view") printf '%s' "$FAKE_PR_JSON" ;;
  "pr checks") printf '%s' "$FAKE_CHECKS" ;;
  *) exit 9 ;;
esac
"""
FAKE_YQ: Final = "#!/usr/bin/env bash\nprintf '%s' \"$FAKE_REQUIRED\"\n"


def check_script() -> str:
    steps: Final = yaml.safe_load(WORKFLOW.read_text())["jobs"]["check-ci-status"]["steps"]
    return next(step["run"] for step in steps if step.get("id") == "check")


def check(workflow: str, bucket: str) -> dict[str, str]:
    return {"name": "job", "workflow": workflow, "bucket": bucket, "link": "https://github.com/o/r/actions/runs/1/job/2"}


def classify(
    tmp_path: Path, check_list: list[dict[str, str]], event: str = "workflow_run", dispatch_pr: str = ""
) -> dict[str, str]:
    for tool, body in (("gh", FAKE_GH), ("yq", FAKE_YQ)):
        path = tmp_path / tool
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    env: Final = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
        "EVENT_NAME": event,
        "DISPATCH_PR": dispatch_pr,
        "HEAD_BRANCH": "upstream-sync/v9.9.9",
        "REPO": "CartoDB/litellm",
        "FAKE_PR_JSON": json.dumps(
            {"number": 139, "headRefName": "upstream-sync/v9.9.9", "labels": [{"name": "upstream-sync"}]}
        ),
        "FAKE_CHECKS": json.dumps(check_list),
        "FAKE_REQUIRED": json.dumps(REQUIRED),
    }
    subprocess.run(["bash", "-c", check_script()], env=env, check=True, capture_output=True)
    return dict(line.split("=", 1) for line in (tmp_path / "out").read_text().splitlines())


@pytest.mark.parametrize("failing", REQUIRED)
def test_every_required_failure_is_fixed_and_upstream_noise_is_ignored(tmp_path: Path, failing: str) -> None:
    outputs: Final = classify(
        tmp_path,
        [
            *(check(wf, "fail" if wf == failing else "pass") for wf in REQUIRED),
            check(UPSTREAM, "fail"),
            check("Upstream cancelled", "cancel"),
        ],
    )
    assert outputs["should-fix"] == "true"
    assert [c["workflow"] for c in json.loads(outputs["blocking-failed"])] == [failing]
    assert outputs["ignored-failed"] == f"{UPSTREAM}, Upstream cancelled"


@pytest.mark.parametrize(
    ("check_list", "should_fix"),
    [
        ([*(check(wf, "pass") for wf in REQUIRED), check(UPSTREAM, "fail")], "false"),
        ([check(REQUIRED[0], "pending"), check(UPSTREAM, "fail")], "false"),
        ([check(REQUIRED[1], "cancel")], "true"),
    ],
    ids=["only-upstream-failing-skips", "pending-required-is-not-a-failure", "cancelled-required-blocks-like-ready"],
)
def test_fixer_runs_only_when_sync_ready_is_blocked(
    tmp_path: Path, check_list: list[dict[str, str]], should_fix: str
) -> None:
    assert classify(tmp_path, check_list)["should-fix"] == should_fix


def test_dispatch_takes_the_pr_number_from_github_not_the_raw_input(tmp_path: Path) -> None:
    outputs: Final = classify(
        tmp_path, [check(REQUIRED[0], "fail")], event="workflow_dispatch", dispatch_pr="upstream-sync/v9.9.9"
    )
    assert outputs["should-fix"] == "true"
    assert (outputs["pr-number"], outputs["branch-name"]) == ("139", "upstream-sync/v9.9.9")
