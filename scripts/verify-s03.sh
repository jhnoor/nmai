#!/usr/bin/env bash
# S03 end-to-end verification script
# Checks: model exists → evaluation scores → weighted score improvement → IMAGE_SIZE → submission build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source venv/bin/activate

PASS=0
FAIL=0

pass() {
    echo "  [PASS] $1"
    ((PASS++))
}

fail() {
    echo "  [FAIL] $1"
    ((FAIL++))
}

echo "=== S03 Verification ==="
echo ""

# -----------------------------------------------------------------------
# Step 1: Check models/best.pt exists and is > 1MB
# -----------------------------------------------------------------------
echo "--- Step 1: Model file check ---"
if [ -f models/best.pt ]; then
    MODEL_SIZE=$(stat -f%z models/best.pt 2>/dev/null || stat -c%s models/best.pt 2>/dev/null)
    if [ "$MODEL_SIZE" -gt 1048576 ]; then
        pass "models/best.pt exists ($(( MODEL_SIZE / 1024 / 1024 ))MB)"
    else
        fail "models/best.pt is too small: ${MODEL_SIZE} bytes"
    fi
else
    fail "models/best.pt does not exist"
fi
echo ""

# -----------------------------------------------------------------------
# Step 2: Run evaluation and capture output
# -----------------------------------------------------------------------
echo "--- Step 2: Evaluation ---"
EVAL_OUTPUT=$(mktemp)
EVAL_EXIT=0
if python scripts/evaluate.py \
    --model models/best.pt \
    --data data/yolo_dataset/dataset.yaml \
    --imgsz 1280 \
    2>&1 | tee "$EVAL_OUTPUT"; then
    pass "evaluate.py exited 0"
else
    EVAL_EXIT=$?
    fail "evaluate.py exited $EVAL_EXIT"
fi
echo ""

# -----------------------------------------------------------------------
# Step 3: Parse weighted score and check > 0.578
# -----------------------------------------------------------------------
echo "--- Step 3: Weighted score check (baseline: 0.578) ---"
WEIGHTED=$(grep -oE 'Weighted Score:.*= [0-9]+\.[0-9]+' "$EVAL_OUTPUT" | grep -oE '[0-9]+\.[0-9]+$' || echo "")
if [ -n "$WEIGHTED" ]; then
    echo "  Weighted score: $WEIGHTED"
    # Use python for float comparison
    if python -c "import sys; sys.exit(0 if float('$WEIGHTED') > 0.578 else 1)"; then
        pass "Weighted score $WEIGHTED > 0.578 (S02 baseline)"
    else
        fail "Weighted score $WEIGHTED <= 0.578 (S02 baseline)"
    fi
else
    fail "Could not parse weighted score from evaluation output"
fi
echo ""

# -----------------------------------------------------------------------
# Step 4: Parse detection mAP and check >= 0.50
# -----------------------------------------------------------------------
echo "--- Step 4: Detection mAP sanity check (>= 0.50) ---"
DET_MAP=$(grep -oE 'Detection mAP@0\.5: *[0-9]+\.[0-9]+' "$EVAL_OUTPUT" | grep -oE '[0-9]+\.[0-9]+$' || echo "")
if [ -n "$DET_MAP" ]; then
    echo "  Detection mAP@0.5: $DET_MAP"
    if python -c "import sys; sys.exit(0 if float('$DET_MAP') >= 0.50 else 1)"; then
        pass "Detection mAP $DET_MAP >= 0.50"
    else
        fail "Detection mAP $DET_MAP < 0.50 — catastrophic regression"
    fi
else
    fail "Could not parse detection mAP from evaluation output"
fi
echo ""

# -----------------------------------------------------------------------
# Step 5: Check src/run.py uses IMAGE_SIZE = 1280
# -----------------------------------------------------------------------
echo "--- Step 5: IMAGE_SIZE check ---"
if grep -q 'IMAGE_SIZE = 1280' src/run.py; then
    pass "src/run.py contains IMAGE_SIZE = 1280"
else
    fail "src/run.py does not contain IMAGE_SIZE = 1280"
fi
echo ""

# -----------------------------------------------------------------------
# Step 6: Build submission zip
# -----------------------------------------------------------------------
echo "--- Step 6: Build submission ---"
if python scripts/build_submission.py 2>&1; then
    pass "build_submission.py exited 0"
else
    fail "build_submission.py exited non-zero"
fi
echo ""

# -----------------------------------------------------------------------
# Step 7: Check submission.zip exists
# -----------------------------------------------------------------------
echo "--- Step 7: Submission zip exists ---"
if [ -f submission.zip ]; then
    ZIP_SIZE=$(stat -f%z submission.zip 2>/dev/null || stat -c%s submission.zip 2>/dev/null)
    pass "submission.zip exists ($(( ZIP_SIZE / 1024 / 1024 ))MB)"
else
    fail "submission.zip does not exist"
fi
echo ""

# -----------------------------------------------------------------------
# Step 8: Failure-path diagnostics
# -----------------------------------------------------------------------
echo "--- Step 8: Failure-path diagnostics ---"

# train.py should show --close-mosaic and --cls in help
if python scripts/train.py --help 2>&1 | grep -q "\-\-close-mosaic"; then
    pass "train.py --help shows --close-mosaic arg"
else
    fail "train.py --help missing --close-mosaic arg"
fi

if python scripts/train.py --help 2>&1 | grep -q "\-\-cls"; then
    pass "train.py --help shows --cls arg"
else
    fail "train.py --help missing --cls arg"
fi

# Evaluation output should contain structured metric lines for machine parsing
if grep -q 'Detection mAP@0.5:' "$EVAL_OUTPUT" && grep -q 'Classification mAP@0.5:' "$EVAL_OUTPUT" && grep -q 'Weighted Score:' "$EVAL_OUTPUT"; then
    pass "Evaluation output contains all three structured metric lines"
else
    fail "Evaluation output missing structured metric lines"
fi
echo ""

# Cleanup
rm -f "$EVAL_OUTPUT"

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo "=== S03 Verification Summary ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    echo "  OVERALL: FAILED"
    exit 1
else
    echo "  OVERALL: ALL PASSED"
    exit 0
fi
