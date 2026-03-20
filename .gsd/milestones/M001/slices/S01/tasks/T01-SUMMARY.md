---
id: T01
parent: S01
milestone: M001
provides:
  - Python 3.12 venv with ultralytics 8.1.0
  - COCO→YOLO converter script
  - Converted YOLO dataset (248 images, 80/20 split)
  - Skeleton verify-s01.sh script
key_files:
  - scripts/convert_coco_to_yolo.py
  - data/yolo_dataset/dataset.yaml
  - scripts/verify-s01.sh
  - .gitignore
key_decisions:
  - ultralytics 8.1.0 installed successfully on Python 3.12.9 — no fallback version needed
patterns_established:
  - Fixed seed 42 for reproducible 80/20 train/val split at image level
  - Symlinked images from main repo using absolute paths (not copies)
  - Bbox normalization with clamping guard (no clamping needed on this dataset)
observability_surfaces:
  - scripts/verify-s01.sh prints pass/fail for label/image counts and dataset.yaml nc assertion
  - Converter prints summary counts (images, annotations, categories, train/val split) to stdout
duration: 8m
verification_result: passed
completed_at: 2026-03-19
blocker_discovered: false
---

# T01: Create Python 3.12 venv and build COCO→YOLO converter

**Created Python 3.12 venv with ultralytics==8.1.0 and COCO→YOLO converter producing 198 train / 50 val images with 22731 normalized bbox annotations and nc=356 dataset.yaml**

## What Happened

Created the Python 3.12 virtual environment at `venv/` using `/opt/homebrew/bin/python3.12`. Installed `ultralytics==8.1.0` and `pycocotools` — both installed cleanly with no version conflicts. Verified ultralytics 8.1.0 imports correctly.

Wrote `scripts/convert_coco_to_yolo.py` (~100 lines) that reads the NorgesGruppen COCO annotations, builds image_id→info and image_id→annotations lookup maps, performs an 80/20 train/val split at the image level with seed=42, converts all bboxes from COCO `[x,y,w,h]` pixel format to YOLO normalized `[x_center,y_center,w,h]` with clamping guards, writes per-image `.txt` label files, symlinks images from the main repo, and generates `dataset.yaml` with nc=356 and all 356 class names.

The converter handled all critical constraints correctly: non-contiguous image IDs (range 1–382), mixed .jpg/.jpeg extensions (210+38), category IDs 0–355. Zero bbox values needed clamping.

Created skeleton `scripts/verify-s01.sh` with converter verification checks (T02/T03 placeholders commented out). Updated `.gitignore` to exclude venv/, data/yolo_dataset/, models/, runs/, __pycache__/, *.pyc.

## Verification

All checks passed:

- `python -c "import ultralytics; print(ultralytics.__version__)"` → `8.1.0`
- `python scripts/convert_coco_to_yolo.py` → exits 0, prints 198 train / 50 val / 22731 annotations
- `ls data/yolo_dataset/labels/train/ | wc -l` → 198
- `ls data/yolo_dataset/labels/val/ | wc -l` → 50
- All 22731 label lines have bbox values in [0,1] range and valid class IDs (0–355)
- Symlinks resolve to real image files in the main repo
- `dataset.yaml` has nc=356 with 356 class names and correct absolute path
- `bash scripts/verify-s01.sh` exits 0

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `source venv/bin/activate && python -c "import ultralytics; print(ultralytics.__version__)"` | 0 | ✅ pass | 1s |
| 2 | `python scripts/convert_coco_to_yolo.py` | 0 | ✅ pass | 2s |
| 3 | `ls data/yolo_dataset/labels/train/ \| wc -l` → 198 | 0 | ✅ pass | <1s |
| 4 | `ls data/yolo_dataset/labels/val/ \| wc -l` → 50 | 0 | ✅ pass | <1s |
| 5 | `python -c "...assert d['nc']==356..."` | 0 | ✅ pass | <1s |
| 6 | `readlink data/yolo_dataset/images/train/img_00001.jpg` → valid path | 0 | ✅ pass | <1s |
| 7 | bbox range check (all 22731 lines in [0,1]) | 0 | ✅ pass | 2s |
| 8 | `bash scripts/verify-s01.sh` | 0 | ✅ pass | 2s |

## Diagnostics

- Run `bash scripts/verify-s01.sh` to re-check all converter outputs in one shot
- Run `python scripts/convert_coco_to_yolo.py` to regenerate the dataset (idempotent — removes existing symlinks)
- Spot-check label files: `head -5 data/yolo_dataset/labels/train/img_00001.txt`
- Check symlink targets: `readlink data/yolo_dataset/images/train/$(ls data/yolo_dataset/images/train/ | head -1)`
- Verify dataset.yaml: `python -c "import yaml; print(yaml.safe_load(open('data/yolo_dataset/dataset.yaml'))['nc'])"`

## Deviations

None — all steps executed as planned.

## Known Issues

None.

## Files Created/Modified

- `venv/` — Python 3.12 virtual environment with ultralytics==8.1.0, pycocotools, and dependencies
- `scripts/convert_coco_to_yolo.py` — COCO→YOLO format converter (108 lines)
- `scripts/verify-s01.sh` — Skeleton S01 verification script with converter checks
- `data/yolo_dataset/` — Converted YOLO dataset (198 train + 50 val images/labels, dataset.yaml)
- `.gitignore` — Excludes venv/, data/yolo_dataset/, models/, runs/, __pycache__/, *.pyc
- `.gsd/milestones/M001/slices/S01/tasks/T01-PLAN.md` — Added Observability Impact section
