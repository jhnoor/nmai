---
estimated_steps: 4
estimated_files: 3
---

# T03: Build evaluation script and run end-to-end pipeline verification

**Slice:** S01 — Data Pipeline & Detection Baseline
**Milestone:** M001

## Description

Write the local evaluation script that computes the competition's weighted mAP metric (`0.7 × detection_mAP + 0.3 × classification_mAP`), then run full end-to-end verification of the entire S01 pipeline. This closes R003 (local evaluation) and proves the complete data→train→eval pipeline works. The baseline scores documented here become the benchmark for S02–S04 to beat.

**Skills to load if available:** none needed — uses pycocotools for mAP computation.

## Steps

1. **Write `scripts/evaluate.py`.** The evaluation script must:
   - Accept CLI args: `--model` (path to .pt file), `--data` (path to dataset.yaml), `--split` (train/val, default: val), `--imgsz` (default: 640), `--conf` (confidence threshold, default: 0.001), `--verbose` (flag to print per-class AP)
   - Load the YOLO model and dataset.yaml to find val image paths
   - Run inference on all val images: `model.predict(source=img_path, conf=conf, imgsz=imgsz, verbose=False)`
   - Convert predictions to COCO format for pycocotools evaluation:
     - Each prediction: `{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}`
     - `image_id` extracted from filename: `img_00042.jpg` → `42` (use `int(path.stem.split('_')[1])`)
     - `bbox` in COCO pixel format `[x, y, w, h]` (convert from YOLO's xyxy output)
     - `category_id` from the model's predicted class index
   - Build ground truth COCO data from the original annotations.json, filtered to val image IDs only
   - Compute **detection mAP** (category-agnostic): Use pycocotools `COCOeval` with `iouType='bbox'` at IoU=0.5. For detection-only evaluation, map all predictions AND ground truth to a single category (e.g., category_id=0) so IoU matching ignores class labels.
   - Compute **classification mAP** (category-aware): Use pycocotools `COCOeval` with `iouType='bbox'` at IoU=0.5 with original category IDs preserved, so a prediction must match both the box location AND the correct category.
   - Compute and print **weighted score**: `0.7 × detection_mAP + 0.3 × classification_mAP`
   - Print a summary like:
     ```
     === Evaluation Results ===
     Val images: 50
     Predictions: 4823
     Detection mAP@0.5: 0.523
     Classification mAP@0.5: 0.187
     Weighted Score: 0.7×0.523 + 0.3×0.187 = 0.422
     ```
   
   **Critical implementation notes for pycocotools:**
   - `COCOeval` expects COCO-format annotation dicts loaded via `COCO()` object
   - For the ground truth, create a COCO object from the annotations.json, filtered to val images
   - For predictions, use `coco_gt.loadRes(predictions_list)` to create a results COCO object
   - Call `cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')` then `cocoEval.params.iouThrs = [0.5]` to evaluate at IoU=0.5 only
   - Access results via `cocoEval.eval['precision']` after calling `evaluate()` and `accumulate()`
   - For detection mAP: temporarily remap all category_ids to 0 in both GT and predictions before creating the COCOeval
   - `cocoEval.stats[1]` is mAP@0.5 (index 1 in the standard 12-metric output), but verify by checking `cocoEval.summarize()` output

2. **Complete `scripts/verify-s01.sh`.** Update the skeleton from T01 to include all slice verification checks:
   ```bash
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
   python -c "from ultralytics import YOLO; m=YOLO('models/best.pt'); print('Model loaded OK:', m.info())"
   
   # Run evaluation
   echo "Running evaluation..."
   python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml
   
   echo "=== S01 Verification PASSED ==="
   ```

3. **Run evaluation and record baseline scores.** Execute:
   ```bash
   source venv/bin/activate
   python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml
   ```
   Record the detection_mAP, classification_mAP, and weighted score. Detection mAP should be > 0.40 for a YOLOv8m trained 50+ epochs on this data. If detection_mAP < 0.40, check for issues: wrong bbox format, wrong nc, training didn't converge.

4. **Run full verification script.** Execute `bash scripts/verify-s01.sh` and confirm it exits 0. This validates the complete pipeline: venv → converter → training → evaluation.

## Must-Haves

- [ ] `scripts/evaluate.py` computes detection_mAP (category-agnostic) and classification_mAP (category-aware) at IoU≥0.5
- [ ] Weighted score formula `0.7×det + 0.3×cls` is correctly implemented and printed
- [ ] Evaluation uses pycocotools COCOeval for mAP computation (competition-standard)
- [ ] `scripts/verify-s01.sh` passes all checks (converter, model, evaluation)
- [ ] Baseline scores documented (detection_mAP, classification_mAP, weighted score)

## Verification

- `python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml` prints detection_mAP > 0.40 and a weighted score
- `bash scripts/verify-s01.sh` exits 0
- Output includes clear breakdown: detection_mAP, classification_mAP, weighted formula, final score

## Inputs

- `models/best.pt` — trained YOLOv8m weights from T02
- `data/yolo_dataset/dataset.yaml` — YOLO dataset config from T01
- `data/yolo_dataset/images/val/` — ~50 val images from T01
- `/Users/jama/code/nmai/norgesgruppen_data/data/train/annotations.json` — original COCO annotations for ground truth
- `scripts/verify-s01.sh` — skeleton from T01 (needs completion)
- `venv/` — Python 3.12 venv from T01

## Expected Output

- `scripts/evaluate.py` — local mAP evaluation script (~120–160 lines)
- `scripts/verify-s01.sh` — complete slice verification script
- Documented baseline scores: detection_mAP, classification_mAP, weighted score (in task summary)
