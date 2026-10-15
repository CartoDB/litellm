import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from carto_feature_test_scope import pytest_args  # noqa: E402


def make_tree(repo: Path, files: Iterable[str], features: Iterable[str], deselect: str = "") -> tuple[Path, Path]:
    for name in files:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).touch()
    manifest = repo / "manifest.yml"
    manifest.write_text("features:\n  - files:\n" + "".join(f"      - {f}\n" for f in features))
    deselect_file = repo / "deselect.txt"
    deselect_file.write_text(deselect)
    return manifest, deselect_file


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_nested_file_maps_to_nearest_mirrored_dir_never_the_root(repo: Path) -> None:
    manifest, deselect = make_tree(
        repo,
        ["tests/test_litellm/llms/databricks/test_x.py", "tests/test_litellm/test_other.py"],
        ["litellm/llms/databricks/chat/transformation.py", "litellm/integrations/unmirrored/foo.py"],
    )
    assert pytest_args(manifest, deselect) == ("tests/test_litellm/llms/databricks",)


def test_top_level_module_maps_to_its_test_file(repo: Path) -> None:
    manifest, deselect = make_tree(repo, ["tests/test_litellm/test_main.py"], ["litellm/main.py", "litellm/router.py"])
    assert pytest_args(manifest, deselect) == ("tests/test_litellm/test_main.py",)


def test_prefers_the_tests_unit_root_and_rewrites_legacy_deselections(repo: Path) -> None:
    manifest, deselect = make_tree(
        repo,
        ["tests/unit/responses/test_a.py", "tests/test_litellm/integrations/test_b.py"],
        ["litellm/responses/handler.py"],
        "# reason\n\ntests/test_litellm/responses/test_a.py::test_x\ntests/unit/responses/test_a.py::test_y\n",
    )
    assert pytest_args(manifest, deselect) == (
        "tests/unit/responses",
        "--deselect=tests/unit/responses/test_a.py::test_x",
        "--deselect=tests/unit/responses/test_a.py::test_y",
    )


def test_no_mirrored_paths_yields_nothing_even_with_deselections(repo: Path) -> None:
    manifest, deselect = make_tree(
        repo, ["tests/test_litellm/other/test_a.py"], ["litellm/llms/x.py"], "tests/test_litellm/a.py::t\n"
    )
    assert pytest_args(manifest, deselect) == ()
