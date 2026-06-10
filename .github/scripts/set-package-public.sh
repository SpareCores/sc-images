#!/usr/bin/env bash
# Ensure an org-scoped GHCR container package is public (idempotent).
# New org packages default to private; inspector hosts pull without registry auth.
#
# Usage: set-package-public.sh <org> <package_name>
# Requires: gh CLI, GH_TOKEN with package admin (GITHUB_TOKEN from publishing workflow).
set -euo pipefail

org="${1:?org required}"
package="${2:?package name required}"

api="orgs/${org}/packages/container/${package}"
status=$(gh api "$api" --jq '.visibility' 2>/dev/null || echo "unknown")
echo "GHCR ${org}/${package} visibility: ${status}"

if [ "$status" = "public" ]; then
  echo "Already public."
  exit 0
fi

echo "Setting visibility to public..."
gh api --method PATCH "$api" --field visibility=public
echo "Package is now public."
