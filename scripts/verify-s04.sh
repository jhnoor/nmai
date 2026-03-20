#!/usr/bin/env bash
# verify-s04.sh — End-to-end verification for S04 (Score Maximization & Final Submissions)
#
# Checks:
#   1. models/best.pt exists and is within size bounds (>100MB, <420MB)
#   2. Evaluation at imgsz=1280 produces detection mAP, classification mAP, weighted score
#   3. submission.zip built successfully by build_submission.py
#   4. src/run.py contains IMAGE_SIZE = 1280
#   5. TTA decision documented in src/run.py
#   6. Failure-path: evaluate.py with nonexistent model exits non-zero

set -euo pipefail

PASS=0
FAIL=0
TOTAL=0

check() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  ✅ PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL: $desc"
        FAIL=$((FAIL + 1))
    fi
}

check_output() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    shift
    local output
    if output=$("$@" 2>&1); then
        echo "  ✅ PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL: $desc"
        echo "     Output: $output"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== S04 Verification ==="
echo ""

# --- Check 1: models/best.pt exists and size bounds ---
echo "[1] Model file checks"
check "models/best.pt exists" test -f models/best.pt

FILESIZE=$(stat -f%z models/best.pt 2>/dev/null || stat -c%s models/best.pt 2>/dev/null || echo 0)
MIN_SIZE=$((100 * 1024 * 1024))   # 100MB
MAX_SIZE=$((420 * 1024 * 1024))   # 420MB

TOTAL=$((TOTAL + 1))
if [ "$FILESIZE" -gt "$MIN_SIZE" ] && [ "$FILESIZE" -lt "$MAX_SIZE" ]; then
    echo "  ✅ PASS: models/best.pt size ($FILESIZE bytes) within bounds (100MB-420MB)"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: models/best.pt size ($FILESIZE bytes) outside bounds (100MB-420MB)"
    FAIL=$((FAIL + 1))
fi

# --- Check 2: Evaluation produces metrics ---
echo ""
echo "[2] Evaluation metrics"
EVAL_OUTPUT=$(venv/bin/python3 scripts/evaluate.py \
    --model models/best.pt \
    --data data/yolo_dataset/dataset.yaml \
    --imgsz 1280 2>&1) || true

TOTAL=$((TOTAL + 1))
if echo "$EVAL_OUTPUT" | grep -q "Detection mAP@0.5:"; then
    DET_MAP=$(echo "$EVAL_OUTPUT" | grep "Detection mAP@0.5:" | awk '{print $NF}')
    echo "  ✅ PASS: Detection mAP@0.5 produced ($DET_MAP)"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: Detection mAP@0.5 not found in evaluation output"
    FAIL=$((FAIL + 1))
fi

TOTAL=$((TOTAL + 1))
if echo "$EVAL_OUTPUT" | grep -q "Classification mAP@0.5:"; then
    CLS_MAP=$(echo "$EVAL_OUTPUT" | grep "Classification mAP@0.5:" | awk '{print $NF}')
    echo "  ✅ PASS: Classification mAP@0.5 produced ($CLS_MAP)"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: Classification mAP@0.5 not found in evaluation output"
    FAIL=$((FAIL + 1))
fi

TOTAL=$((TOTAL + 1))
if echo "$EVAL_OUTPUT" | grep -q "Weighted Score:"; then
    WEIGHTED=$(echo "$EVAL_OUTPUT" | grep "Weighted Score:" | awk '{print $NF}')
    echo "  ✅ PASS: Weighted Score produced ($WEIGHTED)"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: Weighted Score not found in evaluation output"
    FAIL=$((FAIL + 1))
fi

# Check weighted score > 0
TOTAL=$((TOTAL + 1))
if [ -n "${WEIGHTED:-}" ] && python3 -c "import sys; sys.exit(0 if float('$WEIGHTED') > 0 else 1)" 2>/dev/null; then
    echo "  ✅ PASS: Weighted Score ($WEIGHTED) > 0"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: Weighted Score is 0 or missing"
    FAIL=$((FAIL + 1))
fi

# --- Check 3: submission.zip ---
echo ""
echo "[3] Submission build"
BUILD_OUTPUT=$(venv/bin/python3 scripts/build_submission.py 2>&1)
BUILD_EXIT=$?

TOTAL=$((TOTAL + 1))
if [ "$BUILD_EXIT" -eq 0 ]; then
    echo "  ✅ PASS: build_submission.py exits 0"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: build_submission.py exited with $BUILD_EXIT"
    FAIL=$((FAIL + 1))
fi

check "submission.zip exists" test -f submission.zip

TOTAL=$((TOTAL + 1))
if echo "$BUILD_OUTPUT" | grep -q "FAIL"; then
    echo "  ❌ FAIL: build_submission.py reports constraint failures"
    echo "$BUILD_OUTPUT" | grep "FAIL" | sed 's/^/     /'
    FAIL=$((FAIL + 1))
else
    echo "  ✅ PASS: All competition constraints pass"
    PASS=$((PASS + 1))
fi

# --- Check 4: IMAGE_SIZE = 1280 ---
echo ""
echo "[4] Configuration checks"
check "IMAGE_SIZE = 1280 in src/run.py" grep -q "IMAGE_SIZE = 1280" src/run.py

# --- Check 5: TTA decision documented ---
echo ""
echo "[5] TTA decision"
TOTAL=$((TOTAL + 1))
# Check for augment=True on a non-comment line (actual code, not documentation)
if grep -v '^\s*#' src/run.py | grep -q "augment=True"; then
    echo "  ✅ PASS: TTA enabled in src/run.py (augment=True in code)"
    PASS=$((PASS + 1))
elif grep -q "# TTA" src/run.py || grep -q "# augment" src/run.py; then
    echo "  ✅ PASS: TTA decision documented in src/run.py (comment found, TTA not applied)"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: No TTA decision documented in src/run.py (no augment=True and no TTA comment)"
    FAIL=$((FAIL + 1))
fi

# --- Check 6: Failure path ---
echo ""
echo "[6] Failure path check"
TOTAL=$((TOTAL + 1))
# Use a subshell to capture exit code without set -e interfering
FAIL_EXIT=0
venv/bin/python3 scripts/evaluate.py --model /nonexistent.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280 >/dev/null 2>&1 || FAIL_EXIT=$?
if [ "$FAIL_EXIT" -ne 0 ]; then
    echo "  ✅ PASS: Nonexistent model exits non-zero ($FAIL_EXIT)"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: Nonexistent model should exit non-zero but got 0"
    FAIL=$((FAIL + 1))
fi

# --- Summary ---
echo ""
echo "================================"
echo "S04 Verification: $PASS/$TOTAL passed, $FAIL failed"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
