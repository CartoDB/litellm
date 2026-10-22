import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).parent))

from enforce_workflow_allowlist import ALLOWLIST, Workflow, load_keep, workflows_to_disable  # noqa: E402

REPO_ROOT: Final = Path(__file__).parents[2]
KEEP: Final = frozenset({".github/workflows/zizmor.yml"})


def disabled(*workflows: Workflow) -> list[str]:
    return [w.path for w in workflows_to_disable(workflows, KEEP)]


def test_disables_active_upstream_workflows_outside_the_allowlist() -> None:
    assert disabled(
        Workflow(1, ".github/workflows/test-mcp-oauth-e2e.yml", "active"),
        Workflow(2, ".github/workflows/zizmor.yml", "active"),
    ) == [".github/workflows/test-mcp-oauth-e2e.yml"]


def test_never_disables_carto_workflows_even_when_unlisted() -> None:
    assert disabled(Workflow(1, ".github/workflows/carto-brand-new.yml", "active")) == []


def test_skips_workflows_that_are_already_disabled() -> None:
    assert disabled(Workflow(1, ".github/workflows/stale.yml", "disabled_manually")) == []


def test_leaves_github_managed_dynamic_workflows_alone() -> None:
    assert disabled(Workflow(1, "dynamic/dependabot/dependabot-updates", "active")) == []


def test_carto_prefix_must_be_the_file_name_not_a_substring() -> None:
    assert disabled(
        Workflow(1, ".github/workflows/not-carto-ci.yml", "active"),
        Workflow(2, ".github/workflows/cartography.yml", "active"),
    ) == [".github/workflows/cartography.yml", ".github/workflows/not-carto-ci.yml"]


def test_every_allowlisted_workflow_exists() -> None:
    keep: Final = load_keep(REPO_ROOT / ALLOWLIST)
    assert keep
    assert all((REPO_ROOT / path).is_file() for path in keep)
