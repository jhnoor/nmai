#!/usr/bin/env bash
# S02 end-to-end verification script
# Runs: blocked-import check → smoke-test inference → zip build → zip validation → failure-path checks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0

check() {
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  [PASS] $label"
        ((PASS++))
    else
        echo "  [FAIL] $label"
        ((FAIL++))
    fi
}

echo "=== S02 Verification ==="
echo ""

# -----------------------------------------------------------------------
# Step 1: Check for blocked imports in src/run.py
# -----------------------------------------------------------------------
echo "--- Step 1: Blocked imports check ---"
BLOCKED_PATTERN="^import (os|sys|subprocess|socket|shutil|yaml|http|urllib|ftplib|smtplib|ctypes|signal|multiprocessing|threading)"
if grep -qE "$BLOCKED_PATTERN" src/run.py; then
    echo "  [FAIL] Blocked imports found in src/run.py:"
    grep -nE "$BLOCKED_PATTERN" src/run.py
    ((FAIL++))
else
    echo "  [PASS] No blocked imports in src/run.py"
    ((PASS++))
fi
echo ""

# -----------------------------------------------------------------------
# Step 2: Smoke-test inference on 10 images
# -----------------------------------------------------------------------
echo "--- Step 2: Inference smoke test (10 images) ---"

SMOKE_DIR=$(mktemp -d)
SMOKE_OUTPUT="$SMOKE_DIR/predictions.json"
trap 'rm -rf "$SMOKE_DIR"' EXIT

# Symlink first 10 images into smoke dir
IMG_DIR="$REPO_ROOT/norgesgruppen_data/data/train/images"
IMG_SUBSET_DIR="$SMOKE_DIR/images"
mkdir -p "$IMG_SUBSET_DIR"
i=0
for img in "$IMG_DIR"/img_*.jpg; do
    [ -f "$img" ] || continue
    ln -s "$img" "$IMG_SUBSET_DIR/$(basename "$img")"
    ((i++))
    [ "$i" -ge 10 ] && break
done
echo "  Linked $i images for smoke test"

# Activate venv and run inference
source "$REPO_ROOT/venv/bin/activate"
if python src/run.py --input "$IMG_SUBSET_DIR" --output "$SMOKE_OUTPUT" 2>&1; then
    echo "  [PASS] run.py exited 0"
    ((PASS++))
else
    echo "  [FAIL] run.py exited non-zero"
    ((FAIL++))
fi

# Validate predictions JSON schema
if python -c "
import json, sys
d = json.load(open('$SMOKE_OUTPUT'))
assert isinstance(d, list), 'not a list'
assert len(d) > 0, 'empty predictions'
for p in d[:20]:
    assert isinstance(p['image_id'], int), f'image_id not int: {p[\"image_id\"]}'
    assert isinstance(p['category_id'], int), f'category_id not int: {p[\"category_id\"]}'
    assert isinstance(p['bbox'], list) and len(p['bbox']) == 4, f'bbox wrong: {p[\"bbox\"]}'
    assert isinstance(p['score'], float), f'score not float: {p[\"score\"]}'
print(f'  Schema OK: {len(d)} predictions')
" 2>&1; then
    echo "  [PASS] Predictions schema valid"
    ((PASS++))
else
    echo "  [FAIL] Predictions schema invalid"
    ((FAIL++))
fi
echo ""

# -----------------------------------------------------------------------
# Step 3: Build submission zip
# -----------------------------------------------------------------------
echo "--- Step 3: Build submission zip ---"
if python scripts/build_submission.py 2>&1; then
    echo "  [PASS] build_submission.py exited 0"
    ((PASS++))
else
    echo "  [FAIL] build_submission.py exited non-zero"
    ((FAIL++))
fi
echo ""

# -----------------------------------------------------------------------
# Step 4: Validate zip contents
# -----------------------------------------------------------------------
echo "--- Step 4: Validate zip contents ---"

check "submission.zip exists" test -f submission.zip

# Check run.py and best.pt at root, no subdirs
ZIP_LIST=$(python -c "
import zipfile
with zipfile.ZipFile('submission.zip') as zf:
    for name in zf.namelist():
        print(name)
")
if echo "$ZIP_LIST" | grep -qx "run.py"; then
    echo "  [PASS] run.py at zip root"
    ((PASS++))
else
    echo "  [FAIL] run.py NOT at zip root"
    ((FAIL++))
fi

if echo "$ZIP_LIST" | grep -qx "best.pt"; then
    echo "  [PASS] best.pt at zip root"
    ((PASS++))
else
    echo "  [FAIL] best.pt NOT at zip root"
    ((FAIL++))
fi

if echo "$ZIP_LIST" | grep -q "/"; then
    echo "  [FAIL] Zip contains subdirectories"
    ((FAIL++))
else
    echo "  [PASS] No subdirectories in zip"
    ((PASS++))
fi
echo ""

# -----------------------------------------------------------------------
# Step 5: Validate zip size
# -----------------------------------------------------------------------
echo "--- Step 5: Validate zip size ---"
UNCOMPRESSED=$(python -c "
import zipfile
with zipfile.ZipFile('submission.zip') as zf:
    print(sum(e.file_size for e in zf.infolist()))
")
LIMIT=$((420 * 1024 * 1024))
if [ "$UNCOMPRESSED" -le "$LIMIT" ]; then
    echo "  [PASS] Uncompressed size: $((UNCOMPRESSED / 1024 / 1024)) MB (limit: 420 MB)"
    ((PASS++))
else
    echo "  [FAIL] Uncompressed size: $((UNCOMPRESSED / 1024 / 1024)) MB exceeds 420 MB limit"
    ((FAIL++))
fi
echo ""

# -----------------------------------------------------------------------
# Step 6: Failure-path checks
# -----------------------------------------------------------------------
echo "--- Step 6: Failure-path checks ---"

# run.py should exit non-zero on non-existent input
if python src/run.py --input /nonexistent/path --output /tmp/fail_verify.json 2>/dev/null; then
    echo "  [FAIL] run.py should exit non-zero for missing input dir"
    ((FAIL++))
else
    echo "  [PASS] run.py exits non-zero for missing input dir"
    ((PASS++))
fi

# run.py should produce empty array for dir with no images
EMPTY_DIR=$(mktemp -d)
if python src/run.py --input "$EMPTY_DIR" --output "$EMPTY_DIR/empty.json" 2>/dev/null; then
    CONTENT=$(cat "$EMPTY_DIR/empty.json")
    if [ "$CONTENT" = "[]" ]; then
        echo "  [PASS] run.py produces [] for empty image dir"
        ((PASS++))
    else
        echo "  [FAIL] run.py output for empty dir is not []: $CONTENT"
        ((FAIL++))
    fi
else
    echo "  [FAIL] run.py should exit 0 for empty image dir (empty array)"
    ((FAIL++))
fi
rm -rf "$EMPTY_DIR"
echo ""

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo "=== S02 Verification Summary ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    echo "  OVERALL: FAILED"
    exit 1
else
    echo "  OVERALL: ALL PASSED"
    exit 0
fi
