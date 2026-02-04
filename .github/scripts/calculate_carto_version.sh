#!/bin/bash
################################################################################
# CARTO Version Calculator
################################################################################
#
# Calculates the next CARTO semantic version based on conventional commits.
#
# Format: v{upstream}-carto.{MAJOR}.{MINOR}.{PATCH}
#
# Usage:
#   calculate_carto_version.sh [LAST_TAG] [UPSTREAM_VERSION]
#
# Arguments:
#   LAST_TAG          - Previous CARTO release tag (empty for first release)
#   UPSTREAM_VERSION  - Current upstream LiteLLM version (e.g., "1.79.1")
#
# Examples:
#   # First release (analyze all CARTO commits)
#   calculate_carto_version.sh "" "1.79.1"
#
#   # Subsequent release
#   calculate_carto_version.sh "v1.79.1-carto.1.7.1" "1.79.1"
#
#   # After upstream sync (upstream version changes)
#   calculate_carto_version.sh "v1.79.1-carto.1.7.1" "1.79.3"
#
# Output:
#   Prints the next version tag (e.g., "v1.79.1-carto.1.7.2")
#   Exits 1 on error
#
# Note:
#   Uses GitHub API to determine org membership. Requires `gh` CLI authenticated.
#   Commits are counted if the author is a CartoDB org member on GitHub.
#

set -euo pipefail

LAST_TAG="${1:-}"
UPSTREAM_VERSION="${2:-}"

# Repository info (can be overridden via environment)
GITHUB_REPO="${GITHUB_REPOSITORY:-CartoDB/litellm}"
GITHUB_ORG="${GITHUB_ORG:-CartoDB}"
GITHUB_BRANCH="${GITHUB_BRANCH:-carto/main}"

# Validation
if [ -z "$UPSTREAM_VERSION" ]; then
  echo "Error: UPSTREAM_VERSION is required" >&2
  echo "Usage: $0 [LAST_TAG] UPSTREAM_VERSION" >&2
  exit 1
fi

echo "::group::Calculating CARTO version" >&2

# Fetch CartoDB org members (cached for the script duration)
echo "Fetching ${GITHUB_ORG} org members..." >&2
ORG_MEMBERS=$(gh api "orgs/${GITHUB_ORG}/members" --paginate --jq '.[].login' 2>/dev/null | tr '\n' '|' | sed 's/|$//')

if [ -z "$ORG_MEMBERS" ]; then
  echo "Warning: Could not fetch org members, falling back to known bots" >&2
  # Fallback: include known automation accounts
  ORG_MEMBERS="Cartofante|github-actions"
fi

echo "Org members pattern: ${ORG_MEMBERS:0:100}..." >&2

# Function to get commits from GitHub API and filter by org membership
get_carto_commits() {
  local since_tag="$1"
  local commits_json
  local filtered_commits=""

  if [ -z "$since_tag" ]; then
    # First release: get all commits on the branch
    echo "Fetching all commits on ${GITHUB_BRANCH}..." >&2
    commits_json=$(gh api "repos/${GITHUB_REPO}/commits?sha=${GITHUB_BRANCH}&per_page=100" --paginate 2>/dev/null)
  else
    # Get commits since last tag using compare API
    echo "Fetching commits since ${since_tag}..." >&2
    commits_json=$(gh api "repos/${GITHUB_REPO}/compare/${since_tag}...${GITHUB_BRANCH}" --jq '.commits' 2>/dev/null)
  fi

  # Process commits: filter by org membership, exclude merges and upstream syncs
  # Output format: "sha message" for each qualifying commit
  echo "$commits_json" | jq -r '.[] |
    select(.author.login != null) |
    select(.commit.message | test("^Merge"; "i") | not) |
    select(.commit.message | test("sync:|merge upstream"; "i") | not) |
    "\(.author.login)|\(.sha[0:7])|\(.commit.message | split("\n")[0])"
  ' 2>/dev/null | while IFS='|' read -r author sha message; do
    # Check if author is an org member
    if echo "$author" | grep -qE "^(${ORG_MEMBERS})$"; then
      echo "${sha} ${message}"
    fi
  done
}

# Initialize version counters
if [ -z "$LAST_TAG" ]; then
  echo "First CARTO release - analyzing all commits since fork" >&2
  COMMITS=$(get_carto_commits "")
  CURRENT_MAJOR=1
  CURRENT_MINOR=0
  CURRENT_PATCH=0
else
  echo "Analyzing commits since: ${LAST_TAG}" >&2
  COMMITS=$(get_carto_commits "$LAST_TAG")

  # Extract current CARTO version from tag (format: v1.79.1-carto.1.7.1)
  CARTO_VERSION=$(echo "$LAST_TAG" | sed -E 's/.*-carto\.([0-9]+\.[0-9]+\.[0-9]+)/\1/')

  if [ -z "$CARTO_VERSION" ]; then
    echo "Error: Could not extract CARTO version from tag: ${LAST_TAG}" >&2
    exit 1
  fi

  IFS='.' read -r CURRENT_MAJOR CURRENT_MINOR CURRENT_PATCH <<< "$CARTO_VERSION"
  echo "Current CARTO version: ${CURRENT_MAJOR}.${CURRENT_MINOR}.${CURRENT_PATCH}" >&2
fi

# Count commits by type
BREAKING_COUNT=0
FEAT_COUNT=0
FIX_COUNT=0
OTHER_COUNT=0

# Check if there are any commits to process
if [ -z "$COMMITS" ]; then
  echo "No new CARTO commits found" >&2
  echo "Keeping current version (upstream version may have changed)" >&2
else
  # Apply commits in chronological order
  while IFS= read -r commit; do
    if [ -z "$commit" ]; then
      continue
    fi

    # Conventional commit patterns (with optional scope): feat(scope): or feat:
    if echo "$commit" | grep -qiE "BREAKING CHANGE:|breaking(\([^)]*\))?:|major(\([^)]*\))?:"; then
      BREAKING_COUNT=$((BREAKING_COUNT + 1))
      CURRENT_MAJOR=$((CURRENT_MAJOR + 1))
      CURRENT_MINOR=0
      CURRENT_PATCH=0
      echo "  [MAJOR] $commit" >&2
    elif echo "$commit" | grep -qiE "feat(\([^)]*\))?:|feature(\([^)]*\))?:"; then
      FEAT_COUNT=$((FEAT_COUNT + 1))
      CURRENT_MINOR=$((CURRENT_MINOR + 1))
      CURRENT_PATCH=0
      echo "  [MINOR] $commit" >&2
    elif echo "$commit" | grep -qiE "fix(\([^)]*\))?:|bugfix(\([^)]*\))?:"; then
      FIX_COUNT=$((FIX_COUNT + 1))
      CURRENT_PATCH=$((CURRENT_PATCH + 1))
      echo "  [PATCH] $commit" >&2
    else
      OTHER_COUNT=$((OTHER_COUNT + 1))
      # Treat other commits as patches (workflow improvements, etc.)
      CURRENT_PATCH=$((CURRENT_PATCH + 1))
      echo "  [OTHER] $commit" >&2
    fi
  done <<< "$COMMITS"

  echo "" >&2
  echo "Commit summary:" >&2
  echo "  Breaking changes: $BREAKING_COUNT" >&2
  echo "  Features: $FEAT_COUNT" >&2
  echo "  Fixes: $FIX_COUNT" >&2
  echo "  Other: $OTHER_COUNT" >&2
fi

# Construct new version tag
NEW_VERSION="v${UPSTREAM_VERSION}-carto.${CURRENT_MAJOR}.${CURRENT_MINOR}.${CURRENT_PATCH}"

echo "" >&2
echo "New version: ${NEW_VERSION}" >&2
echo "::endgroup::" >&2

# Output the version (stdout for script consumption)
echo "$NEW_VERSION"
