#!/bin/sh
# Pre-commit hook: warn if experiments/ has unregistered changes
# Located at .git/hooks/pre-commit (install manually)

# Get list of staged experiment files
STAGED_EXPERIMENTS=$(git diff --cached --name-only --diff-filter=A | grep '^experiments/experiment_.*\.py$')

if [ -z "$STAGED_EXPERIMENTS" ]; then
    exit 0
fi

echo "=== Pre-registration check ==="
HAS_FAILURE=0

for f in $STAGED_EXPERIMENTS; do
    NAME=$(basename "$f" .py)
    if [ ! -f "pre_registered/${NAME}.json" ]; then
        echo "FAIL: ${NAME} — no pre_registration manifest (pre_registered/${NAME}.json missing)"
        echo "  Create a manifest BEFORE adding experiment files."
        HAS_FAILURE=1
    else
        echo "OK: ${NAME}"
    fi
done

if [ "$HAS_FAILURE" -eq 1 ]; then
    echo ""
    echo "Experiment files must be pre-registered before commit."
    echo "Create a manifest: pre_registered/<name>.json"
    exit 1
fi
