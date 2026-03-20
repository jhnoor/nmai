# S04: Score Maximization & Final Submissions

**Goal:** Best achievable weighted score submitted to leaderboard with TTA evaluated and applied if beneficial.
**Demo:** `bash scripts/verify-s04.sh` passes all checks — final `models/best.pt` is the S03 YOLOv8l model (stripped), evaluation produces weighted score, TTA has been evaluated, and `submission.zip` is built with the best configuration.

## Must-Haves

- S03 YOLOv8l training completed and best.pt copied to `models/` with optimizer stripped (≤420MB)
- Local evaluation of the final model at imgsz=1280 produces a weighted score
- TTA (`augment=True`) evaluated locally — kept if it improves weighted score, removed if not
- `submission.zip` built and validated with the best-scoring configuration
- `scripts/verify-s04.sh` passes all checks

## Proof Level

- This slice proves: final-assembly
- Real runtime required: yes (model inference for evaluation)
- Human/UAT required: yes (upload submission.zip to leaderboard)

## Verification

- `bash scripts/verify-s04.sh` — end-to-end verification covering:
  - models/best.pt exists, is the stripped YOLOv8l model (>100MB, <420MB)
  - evaluation runs and produces detection mAP, classification mAP, weighted score
  - submission.zip built and passes all 7 competition constraints
  - TTA decision documented (augment=True present or absent in run.py with justification)
  - IMAGE_SIZE = 1280 in src/run.py
- Failure-path check: `venv/bin/python3 scripts/evaluate.py --model /nonexistent.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280 2>&1; echo "exit:$?"` — should exit non-zero with a readable error message (not a bare traceback)

## Integration Closure

- Upstream surfaces consumed: `models/best.pt` (S03 training output), `src/run.py`, `scripts/evaluate.py`, `scripts/build_submission.py`
- New wiring introduced in this slice: `augment=True` in `model.predict()` if TTA improves score; `--augment` flag in evaluate.py for A/B comparison
- What remains before the milestone is truly usable end-to-end: human uploads submission.zip to https://app.ainm.no/submit/norgesgruppen-data

## Tasks

- [x] **T01: Complete S03 training handoff — strip optimizer, evaluate, and build baseline submission** `est:30m`
  - Why: S03 launched YOLOv8l training that's running in the background. S04 cannot proceed until the trained model is in models/best.pt with optimizer stripped. This task completes the S03 resume instructions and establishes the baseline score for TTA comparison.
  - Files: `models/best.pt`, `runs/detect/s03_yolov8l_1280/weights/best.pt`
  - Do: Check if training (PID 9132) completed by inspecting results.csv for 80 epochs. If still running, wait. Strip optimizer from the best checkpoint (503MB → ~168MB). Copy to models/best.pt. Run evaluate.py at imgsz=1280. Build submission.zip. Record baseline weighted score.
  - Verify: `ls -la models/best.pt` shows >100MB and <420MB; `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280` produces weighted score; `venv/bin/python3 scripts/build_submission.py` exits 0
  - Done when: models/best.pt is the stripped YOLOv8l model, evaluation weighted score is recorded, and submission.zip is built

- [x] **T02: Add TTA to inference, evaluate impact, build final submission, and write verify-s04.sh** `est:45m`
  - Why: TTA (test-time augmentation) can improve mAP by 1-3% with a single `augment=True` kwarg. This task evaluates TTA's impact, applies the best configuration to run.py, builds the final submission, and writes the slice verification script. Covers requirement R010.
  - Files: `src/run.py`, `scripts/evaluate.py`, `scripts/verify-s04.sh`, `submission.zip`
  - Do: Add `--augment` flag to evaluate.py's run_inference function and CLI args. Run evaluation with and without TTA. If TTA improves weighted score, add `augment=True` to run.py's predict call. Build final submission.zip. Write scripts/verify-s04.sh with all slice verification checks.
  - Verify: `bash scripts/verify-s04.sh` passes all checks
  - Done when: TTA evaluated with measured delta, best config applied to run.py, submission.zip built, verify-s04.sh passes

## Observability / Diagnostics

- **Model file size:** `ls -la models/best.pt` — must be >100MB (real YOLOv8l weights) and <420MB (competition limit). Size outside this range indicates either wrong model or unstripped optimizer.
- **Evaluation metrics output:** `scripts/evaluate.py` prints structured lines: `Detection mAP@50: ...`, `Classification mAP@50: ...`, `Weighted Score: ...`. Absence of any line means the evaluation pipeline broke.
- **Submission validation:** `scripts/build_submission.py` prints pass/fail for each of 7 competition constraints. Any `FAIL` line means the submission won't be accepted.
- **TTA decision trail:** `grep -n 'augment' src/run.py` shows whether TTA is enabled. The verify script checks this and documents the rationale.
- **Training progress:** `wc -l runs/detect/s03_yolov8l_1280/results.csv` shows epoch count. If < 81 (header + 80 data rows), training didn't complete all epochs.
- **Failure visibility:** If evaluation or submission build fails, stderr output from the Python scripts contains the traceback. Exit codes are non-zero on failure.
- **Redaction:** No secrets or API keys are involved in this slice. Model weights are large binary files — don't cat or print their contents.

## Files Likely Touched

- `models/best.pt` — replaced with stripped S03 YOLOv8l model
- `src/run.py` — `augment=True` added to predict call (if TTA helps)
- `scripts/evaluate.py` — `--augment` CLI flag added for A/B testing
- `scripts/verify-s04.sh` — new verification script
- `submission.zip` — rebuilt with final model and configuration
