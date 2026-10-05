#!/usr/bin/env bash
# Run every self-contained tutorial (writes figures to docs/figures and tables to docs/tutorial_outputs).
set -e
cd "$(dirname "$0")/.."
for f in examples/tutorial_*.py; do
    echo "=== $f ==="
    python "$f"
    echo
done
echo "Done. See docs/figures/ and docs/tutorial_outputs/."
