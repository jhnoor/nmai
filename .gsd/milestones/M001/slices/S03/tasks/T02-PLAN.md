---
estimated_steps: 5
estimated_files: 3
---

# T02: Evaluate trained model, rebuild submission, and verify improvement

**Slice:** S03 — Detection Tuning & Classification
**Milestone:** M001

## Description

After the training run launched in T01 completes (~8-12 hours), evaluate the new model against the S02 baseline (det=0.671, cls=0.360, weighted=0.578), rebuild the submission zip, and run the full S03 verification.

**Important context:**
- The venv is at `venv/` — all Python commands use `venv/bin/python3`.
- The project root is `/Users/jama/code/nmai`. All commands run from there.
- T01 launched training with: `--model yolov8l.pt --imgsz 1280 --batch 2 --epochs 80 --device cpu --close-mosaic 0 --cls 1.0 --name s03_yolov8l_1280`
- ultralytics saves outputs to its own project directory (may be the main repo root `/Users/jama/code/nmai`, NOT the worktree). train.py handles this and copies best.pt to `models/best.pt`.
- If training hasn't completed yet, check training status first. If it crashed, investigate logs and potentially re-launch with fallback settings (YOLOv8m instead of YOLOv8l, or reduced epochs).
- S02 baseline scores: detection_mAP=0.671, classification_mAP=0.360, weighted=0.578
- Evaluation uses imgsz=1280 to match training resolution.
- `src/run.py` was already updated to IMAGE_SIZE=1280 in T01.
- `scripts/build_submission.py` copies `src/run.py` and `models/best.pt` into submission.zip. No changes needed to the builder.

## Steps

1. **Check training completion:**
   - Look for the training process (bg_shell list or ps).
   - Check if `models/best.pt` was updated (file modification time should be after T01 started).
   - If training is still running, wait for it. If it crashed, check logs in `runs/detect/s03_yolov8l_1280/` or the ultralytics output dir.
   - If training crashed (OOM, etc.), re-launch with fallback: `venv/bin/python3 scripts/train.py --model yolov8m.pt --imgsz 1280 --batch 4 --epochs 80 --device cpu --close-mosaic 0 --cls 1.0 --name s03_fallback` and wait for completion.

2. **Run evaluation:**
   - Command: `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280`
   - Record: detection_mAP, classification_mAP, weighted score.
   - Compare against S02 baseline: det=0.671, cls=0.360, weighted=0.578.

3. **Rebuild submission.zip:**
   - Command: `venv/bin/python3 scripts/build_submission.py`
   - This picks up the new `models/best.pt` and the updated `src/run.py` (IMAGE_SIZE=1280).
   - Verify it exits 0 with all 7 checks passing.

4. **Run full verification:**
   - Command: `bash scripts/verify-s03.sh`
   - All checks should pass. If weighted score check fails (≤0.578), investigate:
     - Was training long enough? Check final epoch count.
     - Did the model diverge? Check training logs for loss curves.
     - If the improvement is marginal but positive, consider adjusting verify-s03.sh threshold.

5. **Document results:**
   - Record the final scores in a commit message.
   - If any fallback was needed (e.g., YOLOv8m instead of YOLOv8l), note the actual configuration used.

## Must-Haves

- [ ] Training completed successfully (models/best.pt updated with new weights)
- [ ] Evaluation run with imgsz=1280 and scores recorded
- [ ] Weighted score > 0.578 (or documented reason if not achieved)
- [ ] submission.zip rebuilt with new model, passing all 7 competition constraints
- [ ] `bash scripts/verify-s03.sh` passes all checks

## Verification

- `bash scripts/verify-s03.sh` — all checks pass (model loads, weighted > 0.578, submission valid)
- `ls -la models/best.pt` — file updated after T01 training launch
- `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280` — shows weighted score

## Inputs

- `models/best.pt` — newly trained model from T01's training run (or fallback training)
- `scripts/evaluate.py` — evaluation script (unchanged from S01)
- `scripts/build_submission.py` — submission builder (unchanged from S02)
- `scripts/verify-s03.sh` — verification script written in T01
- `src/run.py` — inference script updated with IMAGE_SIZE=1280 in T01

## Expected Output

- `models/best.pt` — confirmed new trained model with improved scores
- `submission.zip` — rebuilt competition submission (~50-87 MB depending on model size)
- Evaluation results: detection_mAP, classification_mAP, weighted score (all documented)
- `bash scripts/verify-s03.sh` — all checks passing

## Observability Impact

- **New signal:** `train.py` now prints "Stripped optimizer" with before/after file sizes when copying best.pt, making it visible when optimizer stripping occurs.
- **Inspection surface:** `ls -la models/best.pt` — YOLOv8l should be ~168MB (stripped), not ~503MB (full checkpoint). If >420MB, optimizer wasn't stripped.
- **Failure visibility:** If `build_submission.py` fails validation check #3 (uncompressed size), the model wasn't stripped. If training crashes, `runs/detect/s03_yolov8l_1280/results.csv` stops updating.
- **Pipeline health:** `bash scripts/verify-s03.sh` covers model loading, evaluation scoring, submission constraints, and diagnostic checks in one command.
