#!/usr/bin/env python3
"""Print the pytest arguments of the CARTO feature gate: mirrored test paths plus deselections.

The gate, the CI fixer and the resolver all run this, so a fix that passes locally passes the gate
"""

import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Final

import yaml

MANIFEST: Final = Path(".github/carto-features.yml")
DESELECT: Final = Path(".github/carto-feature-tests-deselect.txt")
LEGACY_ROOT: Final = "tests/test_litellm"
ROOTS: Final = (Path("tests/unit"), Path(LEGACY_ROOT))


def test_root() -> Path:
    return next(root for root in ROOTS if root.is_dir())


def mirrored_path(feature_file: str, root: Path) -> Path | None:
    parts: Final = PurePosixPath(feature_file).parts
    if parts[0] != "litellm":
        return None
    if len(parts) == 2:
        test_file: Final = root / f"test_{PurePosixPath(parts[1]).stem}.py"
        return test_file if test_file.is_file() else None
    candidates: Final = (root.joinpath(*parts[1:depth]) for depth in range(len(parts) - 1, 1, -1))
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def feature_files(manifest: Path) -> Iterable[str]:
    data: Final = yaml.safe_load(manifest.read_text())
    return (fp for feat in data.get("features", []) for fp in feat.get("files", []))


def deselections(deselect: Path, root: Path) -> tuple[str, ...]:
    lines: Final = (line.strip() for line in deselect.read_text().splitlines())
    return tuple(
        f"--deselect={line.replace(LEGACY_ROOT, str(root), 1) if line.startswith(LEGACY_ROOT + '/') else line}"
        for line in lines
        if line and not line.startswith("#")
    )


def pytest_args(manifest: Path = MANIFEST, deselect: Path = DESELECT) -> tuple[str, ...]:
    root: Final = test_root()
    paths: Final = sorted({str(p) for fp in feature_files(manifest) if (p := mirrored_path(fp, root)) is not None})
    return (*paths, *deselections(deselect, root)) if paths else ()


def main() -> int:
    args: Final = pytest_args()
    if not args:
        print("::error::No test paths derived from .github/carto-features.yml", file=sys.stderr)
        return 1
    print(" ".join(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
