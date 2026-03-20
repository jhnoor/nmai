#!/bin/bash
set -e
echo "=== S01 Verification ==="

# Activate venv
source venv/bin/activate

# Check converter output
TRAIN_LABELS=$(ls data/yolo_dataset/labels/train/ | wc -l | tr -d ' ')
VAL_LABELS=$(ls data/yolo_dataset/labels/val/ | wc -l | tr -d ' ')
TRAIN_IMAGES=$(ls data/yolo_dataset/images/train/ | wc -l | tr -d ' ')
VAL_IMAGES=$(ls data/yolo_dataset/images/val/ | wc -l | tr -d ' ')
echo "Train: $TRAIN_LABELS labels, $TRAIN_IMAGES images"
echo "Val: $VAL_LABELS labels, $VAL_IMAGES images"
[ "$TRAIN_LABELS" -eq "$TRAIN_IMAGES" ] || { echo "FAIL: train label/image count mismatch"; exit 1; }
[ "$VAL_LABELS" -eq "$VAL_IMAGES" ] || { echo "FAIL: val label/image count mismatch"; exit 1; }
TOTAL=$((TRAIN_LABELS + VAL_LABELS))
[ "$TOTAL" -eq 248 ] || { echo "FAIL: expected 248 total, got $TOTAL"; exit 1; }

# Check dataset.yaml
python -c "import yaml; d=yaml.safe_load(open('data/yolo_dataset/dataset.yaml')); assert d['nc']==356, f'nc={d[\"nc\"]}'; print('dataset.yaml: nc=356 OK')"

# Check trained model
[ -f models/best.pt ] || { echo "FAIL: models/best.pt not found"; exit 1; }
SIZE=$(stat -f%z models/best.pt 2>/dev/null || stat --printf="%s" models/best.pt)
[ "$SIZE" -gt 1000000 ] || { echo "FAIL: models/best.pt too small ($SIZE bytes)"; exit 1; }
echo "models/best.pt: ${SIZE} bytes OK"

# Check model loads
python -c "
import functools, numpy as np, torch
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid
_orig = torch.load
@functools.wraps(_orig)
def _p(*a, **kw):
    if 'weights_only' not in kw: kw['weights_only'] = False
    return _orig(*a, **kw)
torch.load = _p
from ultralytics import YOLO
m = YOLO('models/best.pt')
print('Model loaded OK')
"

# Run evaluation
echo "Running evaluation..."
python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml

echo "=== S01 Verification PASSED ==="
