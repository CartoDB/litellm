#!/usr/bin/env bash
# Write the merged CARTO feature PRs, with their changed files, to $1 in `gh pr list --json` shape.
# Paginated REST on purpose: one GraphQL query for every PR with files returns a 502, and the sync agents
# then silently worked without a CARTO feature inventory
set -euo pipefail

OUT="$1"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

ORG_MEMBERS=$(gh api orgs/CartoDB/members --paginate --jq '.[].login' | paste -sd '|')
if [ -z "$ORG_MEMBERS" ]; then
  echo "::error::Could not list CartoDB org members" >&2
  exit 1
fi

MERGED=$(gh api --paginate "repos/${REPO}/pulls?state=closed&per_page=100" \
  --jq '.[] | select(.merged_at != null)
    | {number, title, body, author: {login: .user.login}, mergedAt: .merged_at, labels: [.labels[] | {name}]}' \
  | jq -s '.')
echo "Total merged PRs: $(jq 'length' <<< "$MERGED")" >&2

CARTO=$(jq --arg pattern "$ORG_MEMBERS" '[.[]
  | select(.author.login | test("^(" + $pattern + ")$"; "i"))
  | select([.labels[].name] | any(. == "upstream-sync") | not)
  | select(.title | test("^(sync:|upstream sync:|fix: resolve.*upstream|fix: resolve conflicts|fix\\(ci\\):|merge:)"; "i") | not)
  | select(.title | test("Upstream Sync:"; "i") | not)
]' <<< "$MERGED")

if [ "$(jq 'length' <<< "$CARTO")" -eq 0 ]; then
  echo "::error::No CARTO feature PRs found, refusing to continue without a feature inventory" >&2
  exit 1
fi

FILES_BY_PR=$(jq -r '.[].number' <<< "$CARTO" | while read -r PR; do
  gh api --paginate "repos/${REPO}/pulls/${PR}/files?per_page=100" \
    --jq '.[] | {path: .filename, additions, deletions}' \
    | jq -s --arg pr "$PR" '{key: $pr, value: .}'
done | jq -s 'from_entries')

jq --argjson files "$FILES_BY_PR" '[.[] | ($files[.number | tostring]) as $f
  | . + {
      files: [$f[] | {path}],
      additions: ([$f[].additions] | add // 0),
      deletions: ([$f[].deletions] | add // 0)
    }]' <<< "$CARTO" > "$OUT"
echo "CARTO feature PRs: $(jq 'length' "$OUT")" >&2
