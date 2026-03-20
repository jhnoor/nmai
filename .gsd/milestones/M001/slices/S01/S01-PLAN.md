# S01: Data Pipeline & Detection Baseline

**Goal:** YOLOv8m fine-tuned on the NorgesGruppen shelf dataset with a local evaluation pipeline that produces a weighted detection+classification mAP score on a held-out val set.
**Demo:** `python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml` prints a weighted mAP score with detection_mAP > 0.40.

## Must-Haves

- COCO→YOLO format converter producing `data/yolo_dataset/` with 80/20 train/val split, correct bbox normalization, and `dataset.yaml` with nc=356
- YOLOv8m training script with configurable device (mps/cpu), epochs, image size, and augmentation params
- Trained `models/best.pt` weights from a full training run (50–100 epochs)
- Local evaluation script computing `0.7 × detection_mAP + 0.3 × classification_mAP` at IoU≥0.5
- MPS vs CPU benchmark result documented (5-epoch comparison)
- Python 3.12 venv with ultralytics==8.1.0 pinned

## Proof Level

- This slice proves: contract — local mAP evaluation produces a meaningful detection score
- Real runtime required: yes (model training + inference)
- Human/UAT required: no

## Verification

All verification runs inside the project venv (`source venv/bin/activate`):

- `python scripts/convert_coco_to_yolo.py` completes without error, and:
  - `ls data/yolo_dataset/labels/train/ | wc -l` ≈ 198 (±2)
  - `ls data/yolo_dataset/labels/val/ | wc -l` ≈ 50 (±2)
  - `python -c "import yaml; d=yaml.safe_load(open('data/yolo_dataset/dataset.yaml')); assert d['nc']==356"` passes
- `ls models/best.pt` exists and is > 1MB
- `python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml` outputs a detection_mAP > 0.40 and a weighted score
- `bash scripts/verify-s01.sh` runs all above checks as a single script and exits 0

## Observability / Diagnostics

- Runtime signals: training loss curves and mAP metrics printed by ultralytics during training; evaluation script prints per-metric breakdown
- Inspection surfaces: `scripts/verify-s01.sh` checks all slice outputs in one shot; training run logs in `runs/detect/train*/results.csv`
- Failure visibility: converter prints counts of images/labels processed; training script prints device selection and per-epoch metrics; eval script prints per-class AP when verbose

## Integration Closure

- Upstream surfaces consumed: `norgesgruppen_data/data/train/annotations.json` and `norgesgruppen_data/data/train/images/` (from main repo at `/Users/jama/code/nmai/norgesgruppen_data/data/train/`)
- New wiring introduced in this slice: `data/yolo_dataset/` directory with symlinked images, `models/best.pt` weights, three pipeline scripts
- What remains before the milestone is truly usable end-to-end: S02 builds `run.py` inference script and submission zip; S03 adds classification tuning; S04 does final optimization

## Tasks

- [x] **T01: Create Python 3.12 venv and build COCO→YOLO converter** `est:45m`
  - Why: Everything depends on having the right Python environment and converted training data. The venv retires the Python 3.12 + ultralytics 8.1.0 compatibility risk, and the converter produces the YOLO-format dataset that training and evaluation both consume.
  - Files: `scripts/convert_coco_to_yolo.py`, `data/yolo_dataset/dataset.yaml`, `scripts/verify-s01.sh`, `.gitignore`
  - Do: (1) Create venv with `/opt/homebrew/bin/python3.12 -m venv venv`, install `ultralytics==8.1.0` and `pycocotools`. (2) Write `scripts/convert_coco_to_yolo.py` that reads `/Users/jama/code/nmai/norgesgruppen_data/data/train/annotations.json`, converts COCO bbox `[x,y,w,h]` → YOLO normalized `[x_center,y_center,w,h]`, creates 80/20 stratified train/val split at image level, writes `.txt` label files, symlinks images from main repo, and generates `dataset.yaml` with nc=356. (3) Write skeleton `scripts/verify-s01.sh`. (4) Update `.gitignore` to exclude `venv/`, `data/yolo_dataset/`, `models/`, `runs/`. (5) Run the converter and verify output counts.
  - Verify: `source venv/bin/activate && python scripts/convert_coco_to_yolo.py` succeeds; label counts match expected 80/20 split; spot-check one label file has values in [0,1] range
  - Done when: `data/yolo_dataset/` exists with images/train, images/val, labels/train, labels/val, and dataset.yaml with nc=356 and 356 class names

- [x] **T02: Build training script, benchmark MPS vs CPU, and run full training** `est:2h`
  - Why: Produces the trained model weights (`best.pt`) that S02 needs for inference. The MPS/CPU benchmark retires the training speed risk and informs device choice. This directly delivers R001 (bounding box detection) and R007 (fine-tuning pipeline).
  - Files: `scripts/train.py`, `models/best.pt`
  - Do: (1) Write `scripts/train.py` that loads pretrained YOLOv8m, calls `model.train()` with configurable `--device` (mps/cpu), `--epochs`, `--imgsz` (default 640), `--batch` (default 16), and augmentation params (mosaic, mixup, hsv, flipud, fliplr, scale). (2) Run 5-epoch benchmark on both MPS and CPU, record wall times. (3) Run full training (50–100 epochs) on the faster backend. (4) Copy `runs/detect/train*/weights/best.pt` → `models/best.pt`. Important: use `ultralytics==8.1.0` model name `yolov8m.pt` (NOT `yolo26m.pt`). Start with batch=16 for MPS, reduce to 8 if OOM.
  - Verify: `ls -la models/best.pt` shows file > 1MB; training logs show decreasing loss and rising mAP metrics
  - Done when: `models/best.pt` exists, MPS vs CPU benchmark results documented in task summary, training mAP metrics visible in output

- [x] **T03: Build evaluation script and run end-to-end pipeline verification** `est:45m`
  - Why: Closes R003 (local evaluation script) and proves the entire pipeline works: converted data → trained model → predictions → weighted mAP score. Without this, we can't measure improvement in later slices.
  - Files: `scripts/evaluate.py`, `scripts/verify-s01.sh`
  - Do: (1) Write `scripts/evaluate.py` that loads a YOLO model, runs inference on val images, converts predictions to COCO format, uses pycocotools `COCOeval` to compute detection mAP (category-agnostic, IoU≥0.5) and classification mAP (category-aware, IoU≥0.5), then prints `0.7×det + 0.3×cls` weighted score. Accept `--model` and `--data` args. (2) Complete `scripts/verify-s01.sh` with all slice verification checks. (3) Run evaluation on val set, confirm detection_mAP > 0.40. (4) Document the baseline scores in the task summary for S02 to reference.
  - Verify: `python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml` prints detection_mAP, classification_mAP, and weighted score; `bash scripts/verify-s01.sh` exits 0
  - Done when: Evaluation script produces a weighted mAP score, detection_mAP > 0.40, and verify-s01.sh passes all checks

## Files Likely Touched

- `scripts/convert_coco_to_yolo.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/verify-s01.sh`
- `data/yolo_dataset/dataset.yaml`
- `data/yolo_dataset/images/train/` (symlinks)
- `data/yolo_dataset/images/val/` (symlinks)
- `data/yolo_dataset/labels/train/`
- `data/yolo_dataset/labels/val/`
- `models/best.pt`
- `.gitignore`
