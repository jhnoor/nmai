---
estimated_steps: 5
estimated_files: 5
---

# T01: Create Python 3.12 venv and build COCO→YOLO converter

**Slice:** S01 — Data Pipeline & Detection Baseline
**Milestone:** M001

## Description

Set up the Python 3.12 virtual environment with ultralytics==8.1.0 pinned, then write and run the COCO→YOLO format converter. This is the foundation task — nothing else can proceed without a working venv and converted dataset. The converter reads the COCO annotations.json, converts bounding boxes to YOLO normalized format, creates an 80/20 train/val split at the image level, writes per-image `.txt` label files, symlinks images from the main repo, and generates `dataset.yaml`.

**Skills to load if available:** none needed — this is pure Python scripting with standard libraries.

## Steps

1. **Create venv and install dependencies.** Run `/opt/homebrew/bin/python3.12 -m venv venv` in the worktree root. Activate it: `source venv/bin/activate`. Install: `pip install ultralytics==8.1.0 pycocotools`. Verify: `python -c "import ultralytics; print(ultralytics.__version__)"` prints `8.1.0`. If ultralytics 8.1.0 fails to install on Python 3.12, try `8.1.2` or `8.1.5` as fallback (document which version was used).

2. **Update `.gitignore`.** Add these patterns to the root `.gitignore` (create it if it doesn't exist):
   ```
   venv/
   data/yolo_dataset/
   models/
   runs/
   __pycache__/
   *.pyc
   ```

3. **Write `scripts/convert_coco_to_yolo.py`.** Create the `scripts/` directory. The converter must:
   - Read annotations from `/Users/jama/code/nmai/norgesgruppen_data/data/train/annotations.json`
   - Source images from `/Users/jama/code/nmai/norgesgruppen_data/data/train/images/`
   - Output to `data/yolo_dataset/` (relative to working directory) with structure: `images/train/`, `images/val/`, `labels/train/`, `labels/val/`, `dataset.yaml`
   - Convert COCO bbox `[x, y, w, h]` (top-left pixel coords) → YOLO format `[x_center, y_center, w_norm, h_norm]` (normalized 0–1):
     ```
     x_center = (bbox[0] + bbox[2]/2) / image_width
     y_center = (bbox[1] + bbox[3]/2) / image_height
     w_norm = bbox[2] / image_width
     h_norm = bbox[3] / image_height
     ```
   - Split images 80/20 train/val using `random.shuffle` with a fixed seed (42) for reproducibility. Split at the IMAGE level — all annotations for one image go to the same split.
   - For each image, write a `.txt` label file with one line per annotation: `class_id x_center y_center width height` (space-separated, class_id as int, coords as floats)
   - Label filename must match image stem exactly: `img_00001.jpg` → `img_00001.txt`, `img_00042.jpeg` → `img_00042.txt`
   - Symlink images from the main repo (not copy!) using `os.symlink` with absolute source paths
   - Generate `dataset.yaml` with:
     ```yaml
     path: /Users/jama/code/nmai/.gsd/worktrees/M001/data/yolo_dataset
     train: images/train
     val: images/val
     nc: 356
     names:
       0: "FRØKRISP KNEKKEBRØD ØKOLOGISK 170G BERIT"
       ...
       355: "unknown_product"
     ```
   - Print summary: total images, train/val counts, total annotations, categories

   **Critical constraints:**
   - nc=356 (NOT 357). The actual annotations.json has 356 categories with IDs 0–355. `unknown_product` is at ID 355, not 356. The README's claim of nc=357 is wrong.
   - Image IDs are non-contiguous (range 1–382 with 134 gaps). Must use the `images` list to map `image_id` → `file_name`. Do NOT reconstruct filenames from IDs.
   - Mixed extensions: 210 `.jpg` + 38 `.jpeg`. Preserve original filenames.
   - All normalized bbox values must be in [0, 1] range. Clamp if needed.

4. **Write skeleton `scripts/verify-s01.sh`.** This script will be completed in T03, but create it now with the converter verification checks:
   ```bash
   #!/bin/bash
   set -e
   echo "=== S01 Verification ==="

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

   # Placeholder checks for T02 and T03 (will be filled in later tasks)
   # [ -f models/best.pt ] || { echo "FAIL: models/best.pt not found"; exit 1; }
   # python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml

   echo "=== S01 Verification PASSED ==="
   ```

5. **Run the converter and verify.** Execute `python scripts/convert_coco_to_yolo.py`. Check:
   - Label file counts match expected ~198 train / ~50 val
   - Spot-check: pick one label file, verify all values are in [0,1] range and class_id is an int
   - Symlinks in `images/train/` and `images/val/` point to valid files
   - `dataset.yaml` has nc=356 and correct paths

## Must-Haves

- [ ] Python 3.12 venv with ultralytics==8.1.0 installed and importable
- [ ] `scripts/convert_coco_to_yolo.py` produces correct YOLO-format labels with normalized bboxes
- [ ] 80/20 train/val split at image level, reproducible (fixed seed)
- [ ] Images are symlinked (not copied) from main repo
- [ ] `dataset.yaml` has nc=356, absolute path, and all 356 class names
- [ ] `.gitignore` excludes venv/, data/yolo_dataset/, models/, runs/

## Verification

- `source venv/bin/activate && python -c "import ultralytics; print(ultralytics.__version__)"` prints `8.1.0`
- `python scripts/convert_coco_to_yolo.py` exits 0 and prints summary
- `ls data/yolo_dataset/labels/train/ | wc -l` outputs ~198
- `ls data/yolo_dataset/labels/val/ | wc -l` outputs ~50
- `head -3 data/yolo_dataset/labels/train/*.txt | head -20` shows valid YOLO format lines
- `readlink data/yolo_dataset/images/train/$(ls data/yolo_dataset/images/train/ | head -1)` points to a real file

## Observability Impact

- **Signals changed:** Converter prints summary counts (total images, train/val split, annotations, categories) to stdout on each run. `scripts/verify-s01.sh` prints pass/fail for label counts, image counts, and dataset.yaml nc value.
- **Inspection surface:** `data/yolo_dataset/dataset.yaml` is the primary config artifact — future agents check `nc` and `path` fields. Label `.txt` files can be spot-checked with `head` for bbox sanity. `ls | wc -l` on label/image dirs verifies counts.
- **Failure visibility:** Converter exits non-zero on missing annotations file or image directory. Prints per-image warnings if any bbox value falls outside [0,1] before clamping. verify-s01.sh exits 1 with descriptive message on any count mismatch or yaml assertion failure.

## Inputs

- `/Users/jama/code/nmai/norgesgruppen_data/data/train/annotations.json` — COCO format, 248 images, 22,731 annotations, 356 categories (IDs 0–355). Bbox format: `[x, y, width, height]` in pixels.
- `/Users/jama/code/nmai/norgesgruppen_data/data/train/images/` — 248 shelf images (210 .jpg + 38 .jpeg). Non-contiguous image IDs (range 1–382).
- `/opt/homebrew/bin/python3.12` — Python 3.12.9 binary for venv creation

## Expected Output

- `venv/` — Python 3.12 virtual environment with ultralytics==8.1.0
- `scripts/convert_coco_to_yolo.py` — working COCO→YOLO converter (~80–100 lines)
- `scripts/verify-s01.sh` — skeleton verification script with converter checks
- `data/yolo_dataset/` — complete YOLO dataset directory:
  - `images/train/` — ~198 symlinked images
  - `images/val/` — ~50 symlinked images
  - `labels/train/` — ~198 label .txt files
  - `labels/val/` — ~50 label .txt files
  - `dataset.yaml` — nc=356, absolute path, 356 class names
- `.gitignore` — updated with venv/, data/, models/, runs/ exclusions
