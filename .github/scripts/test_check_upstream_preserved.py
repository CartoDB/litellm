import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from check_upstream_preserved import dropped_upstream_files  # noqa: E402


def run(repo: Path, *args: str) -> str:
    return subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: Path, files: Mapping[str, str], message: str) -> str:
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content)
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", message)
    return run(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run(tmp_path, "init", "-q", "-b", "main")
    run(tmp_path, "config", "user.email", "t@example.com")
    run(tmp_path, "config", "user.name", "t")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def sync_fixture(repo: Path, resolved_content: str) -> tuple[str, str, str]:
    commit(repo, {"litellm/mixed.py": "upstream v1\n", "litellm/upstream_only.py": "v1\n"}, "base")
    run(repo, "checkout", "-q", "-b", "carto")
    carto: Final = commit(repo, {"litellm/mixed.py": "upstream v1\n# CARTO: patch\n"}, "carto")
    run(repo, "checkout", "-q", "main")
    upstream: Final = commit(repo, {"litellm/mixed.py": "upstream v2\n", "litellm/upstream_only.py": "v2\n"}, "up")
    run(repo, "checkout", "-q", "-b", "sync", carto)
    resolved: Final = commit(repo, {"litellm/mixed.py": resolved_content, "litellm/upstream_only.py": "v2\n"}, "res")
    return upstream, carto, resolved


def test_flags_file_resolved_to_the_carto_copy(repo: Path) -> None:
    upstream, carto, resolved = sync_fixture(repo, "upstream v1\n# CARTO: patch\n")
    assert dropped_upstream_files(upstream, carto, resolved) == ("litellm/mixed.py",)


def test_passes_when_carto_block_is_laid_on_upstream(repo: Path) -> None:
    upstream, carto, resolved = sync_fixture(repo, "upstream v2\n# CARTO: patch\n")
    assert dropped_upstream_files(upstream, carto, resolved) == ()


def test_ignores_build_artifacts(repo: Path) -> None:
    artifact: Final = "litellm/proxy/_experimental/out/index.html"
    commit(repo, {artifact: "v1\n"}, "base")
    run(repo, "checkout", "-q", "-b", "carto")
    carto: Final = commit(repo, {artifact: "carto\n"}, "carto")
    run(repo, "checkout", "-q", "main")
    upstream: Final = commit(repo, {artifact: "v2\n"}, "up")
    assert dropped_upstream_files(upstream, carto, carto) == ()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
