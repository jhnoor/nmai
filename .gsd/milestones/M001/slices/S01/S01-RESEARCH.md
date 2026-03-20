# S01: Data Pipeline & Detection Baseline — Research

**Date:** 2026-03-19
**Depth:** Targeted — known technology (ultralytics YOLO), new dataset, moderate integration complexity.

## Summary

S01 must deliver four scripts and a trained model: (1) COCO→YOLO format conversion with train/val split, (2) a training script for YOLOv8 fine-tuning, (3) a local evaluation script computing the competition's weighted mAP, and (4) a trained `best.pt` model. The dataset is 248 images (not 210 as the roadmap states) with 22,731 annotations across 356 categories (IDs 0–355). The data lives in the main repo at `/Users/jama/code/nmai/norgesgruppen_data/data/train/` — the worktree has empty placeholder dirs due to `.gitignore`, so scripts must reference this path or the conversion script must populate the worktree's `data/yolo_dataset/` directory.

The primary risks are: (a) ultralytics 8.1.0 may not install cleanly on Python 3.13 (only 3.12 and 3.13 are available — 3.12 is safer), (b) the nc value discrepancy between README (nc=357) and actual data (356 categories, IDs 0–355), and (c) MPS vs CPU training speed on this small dataset. The conversion is straightforward COCO→YOLO math, but there's a filename extension mix (.jpg and .jpeg) that the converter must handle correctly by preserving original filenames.

## Recommendation

**Write a custom COCO→YOLO converter** rather than using `ultralytics.data.converter.convert_coco` — the built-in converter is designed for segmentation and standard COCO layout, not our flat single-directory structure. A custom 80-line script gives full control over the train/val split strategy and handles our specific data quirks (mixed extensions, non-contiguous image IDs, 356-class mapping).

**Use Python 3.12** for the venv — ultralytics 8.1.0 was released in early 2024 and is unlikely to support Python 3.13. Python 3.12 is available at `/opt/homebrew/bin/python3.12`.

**Set nc=356** — the actual training data has 356 categories with IDs 0–355. The README's claim of nc=357 and "unknown_product at ID 356" contradicts the annotations.json which has `unknown_product` at ID 355. Use what the data actually contains.

**Start with YOLOv8m** — medium model balances accuracy and training speed on this small dataset. Larger models (l, x) can be tested in S03.

**80/20 stratified split** — with 248 images, an 80/20 split gives ~198 train / 50 val. Stratify by ensuring each split has proportional representation of images from different density ranges (14–235 annotations per image).

## Implementation Landscape

### Key Files

- `norgesgruppen_data/data/train/annotations.json` — COCO format source, 248 images, 22,731 annotations, 356 categories (0–355). Bbox format: `[x, y, width, height]` in pixels. Mixed `.jpg`/`.jpeg` extensions (210 jpg, 38 jpeg). Non-contiguous image IDs (range 1–382 with 134 gaps). All 248 annotated filenames match files on disk.
- `norgesgruppen_data/data/train/images/` — 248 shelf images, varying resolutions (481×399 to 5712×4624, typical ~2000×1500 and ~4032×3024). Located in main repo at `/Users/jama/code/nmai/norgesgruppen_data/data/train/images/`.
- `scripts/convert_coco_to_yolo.py` — TO CREATE. Reads annotations.json, writes YOLO label `.txt` files and `dataset.yaml`. Must handle: COCO `[x,y,w,h]` → YOLO `[x_center,y_center,w,h]` normalized, mixed file extensions, train/val split.
- `scripts/train.py` — TO CREATE. Loads pretrained YOLOv8m, fine-tunes with `dataset.yaml`, configurable device (mps/cpu), epochs, imgsz, augmentation params.
- `scripts/evaluate.py` — TO CREATE. Uses pycocotools COCOeval to compute detection mAP and classification mAP at IoU≥0.5, then computes `0.7 × det_mAP + 0.3 × cls_mAP`.
- `data/yolo_dataset/` — TO CREATE by converter. Structure: `images/train/`, `images/val/`, `labels/train/`, `labels/val/`, `dataset.yaml`.
- `models/best.pt` — TO CREATE by training. YOLOv8m weights fine-tuned on this dataset.

### Data Details for Converter

The COCO→YOLO conversion formula per annotation:
```
x_center = (bbox[0] + bbox[2]/2) / image_width
y_center = (bbox[1] + bbox[3]/2) / image_height
w_norm = bbox[2] / image_width
h_norm = bbox[3] / image_height
```

YOLO label file: one `.txt` per image, same stem as image file. Each line: `class_id x_center y_center width height` (space-separated, all floats normalized 0–1 except class_id which is int).

The `dataset.yaml` structure:
```yaml
path: /absolute/path/to/yolo_dataset
train: images/train
val: images/val
nc: 356
names:
  0: "FRØKRISP KNEKKEBRØD ØKOLOGISK 170G BERIT"
  1: "COFFEE MATE 180G  NESTLE"
  ...
  355: "unknown_product"
```

### Category Distribution (informs augmentation strategy)

| Bin | Categories | Total Annotations |
|-----|-----------|-------------------|
| 1 annotation | 41 | 41 |
| 2–4 | 33 | 95 |
| 5–9 | 36 | 240 |
| 10–19 | 48 | 693 |
| 20–49 | 56 | 1,780 |
| 50–99 | 51 | 3,489 |
| 100–199 | 66 | 9,234 |
| 200–499 | 25 | 7,159 |

74 categories have <5 examples — classification mAP for these will be near-zero. This is expected and acceptable given the 70/30 detection/classification weighting.

### Build Order

1. **Environment setup** — create venv with Python 3.12, install `ultralytics==8.1.0` and dependencies. This unblocks everything and retires the Python compatibility risk immediately.

2. **COCO→YOLO converter** (`scripts/convert_coco_to_yolo.py`) — this is the foundation. No training without converted data. Must produce `data/yolo_dataset/` with proper structure. Verify by checking label file count matches image count, spot-checking a few bbox conversions, confirming dataset.yaml is valid.

3. **Training script** (`scripts/train.py`) — depends on converted data. First run benchmarks MPS vs CPU (run 5 epochs on each, compare wall time). This retires the MPS speed risk. Use the faster backend for the full training run.

4. **Evaluation script** (`scripts/evaluate.py`) — can be built in parallel with training (uses pycocotools, already installed). Needs a model to produce predictions for testing, but the script structure is independent. Verify by running on val set predictions from the trained model.

5. **Full training run** — after MPS/CPU benchmark, run full training (50–100 epochs) with the faster backend. Save `best.pt`.

### Verification Approach

1. **Converter verification:**
   - `ls data/yolo_dataset/labels/train/ | wc -l` should equal ~198 (80% of 248)
   - `ls data/yolo_dataset/labels/val/ | wc -l` should equal ~50 (20% of 248)
   - Each `.txt` label file has one line per annotation, all values in [0,1] range
   - Spot-check: pick an annotation, manually compute YOLO coords, compare to label file
   - `dataset.yaml` has nc=356, correct paths, all 356 class names

2. **Training verification:**
   - Training script starts and produces loss output without errors
   - MPS vs CPU benchmark: 5 epochs each, record wall time
   - After full training: `best.pt` exists in runs directory, training mAP metrics visible

3. **Evaluation verification:**
   - Evaluation script loads model, runs inference on val images
   - Produces per-class AP values and weighted mAP score
   - Detection mAP (category-agnostic) should be >0.40 for a reasonable model
   - Weighted score `0.7×det + 0.3×cls` is printed

4. **End-to-end:**
   - `python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml` produces a score

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| mAP evaluation | pycocotools `COCOeval` | Already installed (v2.0.11), competition-standard COCO evaluation |
| YOLO training pipeline | ultralytics `YOLO.train()` | Handles augmentation, lr scheduling, checkpointing, metrics — all battle-tested |
| Image augmentation | Built into ultralytics (mosaic, mixup, hsv, flip, scale) | No need for separate albumentations pipeline for baseline |

## Constraints

- **Python 3.12 required** — only 3.12 (`/opt/homebrew/bin/python3.12`) and 3.13 available locally. ultralytics 8.1.0 likely incompatible with 3.13. Must create a venv with 3.12.
- **ultralytics must be pinned to 8.1.0** — sandbox uses this version. Weight format changes between versions cause segfaults (D005).
- **Data lives in main repo** — worktree has empty data dirs. Scripts must use absolute paths to `/Users/jama/code/nmai/norgesgruppen_data/data/train/` OR the converter should symlink/copy images into the worktree's `data/yolo_dataset/`. Symlinks are strongly preferred (248 large images = ~2GB).
- **Mixed file extensions** — 210 `.jpg` + 38 `.jpeg` files. Label filenames must match image stems exactly (YOLO convention: `img_00001.jpg` → `img_00001.txt`).
- **nc=356, NOT 357** — the README claims nc=357 but actual annotations.json has 356 categories (IDs 0–355). `unknown_product` is at ID 355, not 356. Use nc=356.
- **`yaml` is a blocked import in sandbox** — but dataset.yaml is only needed at training time (local), not inference time. No constraint for S01, but `run.py` (S02) cannot use yaml. Use `json` for any runtime config.
- **No network in sandbox** — model weights must be bundled. YOLOv8m.pt is ~50MB, well within 420MB limit.

## Common Pitfalls

- **COCO bbox off-by-one** — COCO bbox is `[x_min, y_min, width, height]` (top-left origin). Some implementations confuse this with `[x_min, y_min, x_max, y_max]`. The annotations.json uses the standard COCO format (verified: first annotation `[141, 49, 169, 152]` on 2000×1500 image gives reasonable normalized values).
- **Image ID ≠ filename number** — image IDs range from 1–382 with 134 gaps. Must use the `images` list to map `image_id` → `file_name`, not reconstruct filenames from IDs.
- **Train/val data leakage** — the split must be at the image level, not the annotation level. All annotations for one image go to the same split.
- **MPS memory issues** — MPS on M1 Max can OOM on large batches. Start with batch=16 and increase if stable. If MPS is problematic, CPU with batch=8 on 64GB RAM will work.
- **ultralytics 8.1.0 model names** — this version uses `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, etc. — NOT `yolo26n.pt` (that's a newer version naming). Ensure scripts use `yolov8m.pt`.

## Open Risks

- **ultralytics 8.1.0 + Python 3.12 compatibility** — 8.1.0 was released Jan 2024, Python 3.12 was released Oct 2023. Should work but untested. If install fails, try 8.1.2 or 8.1.5 as closest alternatives.
- **MPS training may be slower than CPU for 248 images** — small dataset means data loading overhead dominates, and MPS kernel launch overhead may exceed GPU speedup. The 5-epoch benchmark will retire this risk.
- **Val set representativeness** — with only 248 images and 356 categories, the val set (~50 images) won't cover all categories. Local mAP may not correlate well with leaderboard. Accept this — it's inherent to the small dataset.

## Forward Intelligence (for S02)

- The trained model `best.pt` will be loaded in `run.py` using `ultralytics.YOLO("best.pt")`. S02 must use `pathlib` for all file ops and cannot import `os`, `yaml`, or `sys`.
- Model output from `model.predict()` returns `Results` objects with `.boxes.xyxy`, `.boxes.conf`, `.boxes.cls` attributes. S02's `run.py` needs to convert these to COCO format `[x, y, w, h]` for the output JSON.
- The `image_id` in output must be extracted from filename: `img_00042.jpg` → `42`. Use `int(path.stem.split('_')[1])`.
- Evaluation script's mAP computation approach (pycocotools) should match what the competition uses, giving confidence that local scores approximate leaderboard scores.
