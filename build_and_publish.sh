#!/usr/bin/env bash
# build_and_publish.sh -- Build and publish RAVANA packages to PyPI
# Usage: ./build_and_publish.sh [test|prod] [ml|grace|chat|all]
#   test  -> TestPyPI (default)
#   prod  -> Production PyPI
#   ml|grace|chat|all -> which package(s) to publish

set -euo pipefail

TARGET="${1:-test}"
PACKAGE="${2:-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================="
echo "RAVANA PyPI Build & Publish"
echo "Target: $TARGET"
echo "Package: $PACKAGE"
echo "==============================================="

# Check for token
if [[ "$TARGET" == "prod" ]]; then
    if [[ -z "${PYPI_TOKEN:-}" ]]; then
        echo "ERROR: PYPI_TOKEN environment variable required for production"
        echo "  export PYPI_TOKEN='***'"
        exit 1
    fi
    REPO_URL="https://upload.pypi.org/legacy/"
    REPO_NAME="pypi"
else
    if [[ -z "${TESTPYPI_TOKEN:-}" ]]; then
        echo "ERROR: TESTPYPI_TOKEN environment variable required for TestPyPI"
        echo "  export TESTPYPI_TOKEN='***'"
        exit 1
    fi
    REPO_URL="https://test.pypi.org/legacy/"
    REPO_NAME="testpypi"
fi

clean_build() {
    local dir="$1"
    find "$dir" -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
    find "$dir" -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
    find "$dir" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
}

build_package() {
    local dir="$1"
    local name="$2"
    echo ""
    echo "--- Building $name ---"
    cd "$dir"
    clean_build "$dir"
    python -m build --wheel --sdist
    echo "Built: $(ls dist/)"
}

publish_package() {
    local dir="$1"
    local name="$2"
    echo ""
    echo "--- Publishing $name to $TARGET ---"
    cd "$dir"
    if [[ "$TARGET" == "prod" ]]; then
        python -m twine upload --repository-url "$REPO_URL" -u "__token__" -p "$PYPI_TOKEN" dist/*
    else
        python -m twine upload --repository-url "$REPO_URL" -u "__token__" -p "$TESTPYPI_TOKEN" dist/*
    fi
}

# Main execution
case "$PACKAGE" in
    all)
        build_package "$ROOT_DIR/ravana_ml" "ravana-ml"
        build_package "$ROOT_DIR/ravana-v2" "ravana-grace"
        build_package "$ROOT_DIR/ravana" "ravana-chat"
        publish_package "$ROOT_DIR/ravana_ml" "ravana-ml"
        publish_package "$ROOT_DIR/ravana-v2" "ravana-grace"
        publish_package "$ROOT_DIR/ravana" "ravana-chat"
        ;;
    ml)
        build_package "$ROOT_DIR/ravana_ml" "ravana-ml"
        publish_package "$ROOT_DIR/ravana_ml" "ravana-ml"
        ;;
    grace)
        build_package "$ROOT_DIR/ravana-v2" "ravana-grace"
        publish_package "$ROOT_DIR/ravana-v2" "ravana-grace"
        ;;
    chat)
        build_package "$ROOT_DIR/ravana" "ravana-chat"
        publish_package "$ROOT_DIR/ravana" "ravana-chat"
        ;;
    *)
        echo "Unknown package: $PACKAGE"
        echo "Usage: $0 [test|prod] [all|ml|grace|chat]"
        exit 1
        ;;
esac

echo ""
echo "==============================================="
echo "Done! Check https://${REPO_NAME%.org}.org or https://pypi.org"
echo "==============================================="