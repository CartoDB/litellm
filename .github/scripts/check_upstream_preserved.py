#!/usr/bin/env python3
"""Fail an upstream-sync merge that kept CARTO's copy of a file both sides changed.

For every path changed on both sides since the last synced upstream point, the
resolved file must not be byte-identical to carto/main while upstream's version
differs: that means upstream's changes to the file were dropped wholesale
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from typing import Final

CHECKED_PREFIXES: Final = ("litellm/", "enterprise/", "litellm-proxy-extras/", "tests/")
IGNORED_SEGMENTS: Final = ("/_experimental/out/",)


def git(*args: str) -> str:
    return subprocess.run(("git", *args), check=True, capture_output=True, text=True).stdout


def changed_paths(base: str, head: str) -> frozenset[str]:
    return frozenset(git("diff", "--name-only", "--no-renames", base, head).splitlines())


def blobs(ref: str, paths: Iterable[str]) -> Mapping[str, str]:
    path_list: Final = sorted(paths)
    if not path_list:
        return {}
    out: Final = git("ls-tree", "-r", "--full-tree", ref, "--", *path_list)
    return {line.split("\t", 1)[1]: line.split()[2] for line in out.splitlines()}


def is_checked(path: str) -> bool:
    return path.startswith(CHECKED_PREFIXES) and not any(segment in path for segment in IGNORED_SEGMENTS)


def dropped_upstream_files(upstream: str, carto: str, resolved: str) -> tuple[str, ...]:
    base: Final = git("merge-base", upstream, carto).strip()
    both_changed: Final = frozenset(
        path for path in changed_paths(base, upstream) & changed_paths(base, carto) if is_checked(path)
    )
    upstream_blobs: Final = blobs(upstream, both_changed)
    carto_blobs: Final = blobs(carto, both_changed)
    resolved_blobs: Final = blobs(resolved, both_changed)
    return tuple(
        sorted(
            path
            for path in both_changed
            if resolved_blobs.get(path) == carto_blobs.get(path) and upstream_blobs.get(path) != carto_blobs.get(path)
        )
    )


def report(dropped: tuple[str, ...], upstream: str) -> str:
    if not dropped:
        return "### Upstream preserved\n\nNo file kept carto/main's copy over upstream changes\n"
    rows: Final = "\n".join(f"- `{path}`" for path in dropped)
    return (
        "### Upstream changes dropped\n\n"
        f"These files changed on both sides, yet the resolution is identical to carto/main and ignores `{upstream}`. "
        "Re-apply upstream's version and lay the `# CARTO` blocks back on top\n\n"
        f"{rows}\n"
    )


def main() -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="upstream tag or commit being synced")
    parser.add_argument("--carto", required=True, help="carto/main before the sync")
    parser.add_argument("--resolved", default="HEAD", help="resolved sync branch")
    args: Final = parser.parse_args()

    dropped: Final = dropped_upstream_files(args.upstream, args.carto, args.resolved)
    text: Final = report(dropped, args.upstream)
    print(text)
    summary_path: Final = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(text)
    return 1 if dropped else 0


if __name__ == "__main__":
    sys.exit(main())
