#!/usr/bin/env bash
# Refresh the published (GitHub Pages) explorer from the local data repository.
#
# Pull fresh raw data first (see docs/getting-started.md, scripts/pull_covariates.py).
# This script rebuilds the explorer index only if raw data changed, rewrites the
# thinned static export in docs/explorer/data/, and leaves committing to you:
# the Documentation workflow publishes it on the next push to main.
#
#   scripts/update_published_explorer.sh            # index + export
#   scripts/update_published_explorer.sh --preview  # also serve the published copy locally
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
PORT="${PORT:-8000}"

"$PYTHON" scripts/explore_covariates.py --data-root "$DATA_ROOT" index
"$PYTHON" scripts/explore_covariates.py --data-root "$DATA_ROOT" export --out docs/explorer/data

du -sh docs/explorer/data
git status --short docs/explorer/data

if [[ "${1:-}" == "--preview" ]]; then
  # Assemble the page exactly as the Documentation workflow does, outside the repo.
  preview="$(mktemp -d)"
  cp -R docs/explorer/data "$preview/"
  cp src/tapestry/explorer/static/index.html src/tapestry/explorer/static/app.js \
     src/tapestry/explorer/static/style.css "$preview/"
  echo "Published copy: http://127.0.0.1:$PORT/ (Ctrl-C to stop)"
  "$PYTHON" -m http.server "$PORT" --bind 127.0.0.1 --directory "$preview"
else
  echo 'Publish: git add docs/explorer/data && git commit -m "Update published explorer data" && git push'
fi
