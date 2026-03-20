# S03: Detection Tuning & Classification

**Goal:** Train an improved YOLOv8 model with larger architecture (YOLOv8l), higher resolution (imgsz=1280), continuous mosaic augmentation (close_mosaic=0), and increased classification loss weight (cls=1.0), achieving a weighted score above the S02 baseline of 0.578.
**Demo:** `bash scripts/verify-s03.sh` passes all checks: new model loads, weighted score > 0.578, submission.zip rebuilt and valid.

## Must-Haves

- `scripts/train.py` supports `--close-mosaic` and `--cls` CLI arguments for tuning
- Training run completed with YOLOv8l at imgsz=1280, close_mosaic=0, cls=1.0
- `src/run.py` uses IMAGE_SIZE=1280 to match training resolution
- `models/best.pt` contains the new trained model
- Local evaluation shows weighted score > 0.578 (S02 baseline: det=0.671, cls=0.360, weighted=0.578)
- `submission.zip` rebuilt with the new model and updated run.py, passing all competition constraints
- `scripts/verify-s03.sh` validates all of the above

## Proof Level

- This slice proves: operational (real training + real evaluation on real data)
- Real runtime required: yes (CPU training 8-12h, local evaluation ~5min)
- Human/UAT required: no (leaderboard upload is a follow-up)

## Verification

- `bash scripts/verify-s03.sh` — end-to-end verification covering:
  - New model loads successfully
  - Evaluation produces detection mAP, classification mAP, and weighted score
  - Weighted score > 0.578 (improvement over S02 baseline)
  - `submission.zip` rebuilt and passes all 7 competition constraints
  - `src/run.py` uses IMAGE_SIZE=1280
  - Diagnostic: train.py --help exposes --close-mosaic and --cls args (failure-path: missing args = training misconfigured)
  - Diagnostic: evaluation output contains all three structured metric lines (failure-path: if absent, score parsing fails silently)

## Observability / Diagnostics

- Runtime signals: train.py prints epoch progress, loss curves, and final metrics to stdout; evaluate.py prints per-metric mAP scores
- Inspection surfaces: `scripts/verify-s03.sh` (single command to validate entire slice), `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280` (direct evaluation)
- Failure visibility: If training diverges — val mAP will be lower than baseline. If OOM — training exits with error, best.pt not updated. verify-s03.sh reports which specific check failed.

## Integration Closure

- Upstream surfaces consumed: `models/best.pt` (S01 trained weights, overwritten by this slice), `src/run.py` (S02 inference script, updated with new IMAGE_SIZE), `scripts/build_submission.py` (S02 zip builder, unchanged), `scripts/evaluate.py` (S01 evaluation script, unchanged)
- New wiring introduced in this slice: `--close-mosaic` and `--cls` args in train.py; IMAGE_SIZE=1280 in run.py
- What remains before the milestone is truly usable end-to-end: S04 (TTA/augmentation tuning, final submission upload)

## Tasks

- [x] **T01: Add training tuning args and launch improved training run** `est:30m`
  - Why: train.py lacks CLI args for close_mosaic and cls loss weight, which are the key tuning levers identified in research. The training run itself takes 8-12h and must be started ASAP.
  - Files: `scripts/train.py`, `src/run.py`, `scripts/verify-s03.sh`
  - Do: Add `--close-mosaic` (int, default=10) and `--cls` (float, default=0.5) args to train.py and pass them to model.train(). Update `src/run.py` IMAGE_SIZE from 640 to 1280. Write `scripts/verify-s03.sh` that checks model loading, evaluation scores, and submission validity. Launch training: `venv/bin/python3 scripts/train.py --model yolov8l.pt --imgsz 1280 --batch 2 --epochs 80 --device cpu --close-mosaic 0 --cls 1.0 --name s03_yolov8l_1280`. If YOLOv8l OOMs at batch=2, fall back to YOLOv8m with same args.
  - Verify: `venv/bin/python3 scripts/train.py --help` shows new args; `grep IMAGE_SIZE src/run.py` shows 1280; `bash scripts/verify-s03.sh` exists and is executable (full pass deferred to T02 after training completes)
  - Done when: train.py has --close-mosaic and --cls args, run.py uses IMAGE_SIZE=1280, verify-s03.sh is written, training is running in background

- [x] **T02: Evaluate trained model, rebuild submission, and verify improvement** `est:20m`
  - Why: Training from T01 produces a new best.pt that must be evaluated against the S02 baseline, and the submission zip must be rebuilt with the new model + updated run.py.
  - Files: `models/best.pt`, `submission.zip`, `scripts/verify-s03.sh`
  - Do: Confirm training completed and models/best.pt was updated (check file modification time). Run `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280` and record det_mAP, cls_mAP, weighted. If weighted > 0.578, proceed. Run `venv/bin/python3 scripts/build_submission.py` to rebuild submission.zip. Run `bash scripts/verify-s03.sh` end-to-end. If training produced a model worse than baseline (weighted ≤ 0.578), investigate and document — the slice verification threshold may need to be adjusted to reflect achievable improvement.
  - Verify: `bash scripts/verify-s03.sh` passes all checks
  - Done when: verify-s03.sh passes all checks, weighted score documented, submission.zip rebuilt with new model

## Files Likely Touched

- `scripts/train.py` — add --close-mosaic and --cls CLI args
- `src/run.py` — update IMAGE_SIZE from 640 to 1280
- `scripts/verify-s03.sh` — new verification script for S03
- `models/best.pt` — overwritten by new training run
- `submission.zip` — rebuilt with new model
