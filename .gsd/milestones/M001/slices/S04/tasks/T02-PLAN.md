---
estimated_steps: 5
estimated_files: 4
---

# T02: Add TTA to inference, evaluate impact, build final submission, and write verify-s04.sh

**Slice:** S04 — Score Maximization & Final Submissions
**Milestone:** M001

## Description

Test-time augmentation (TTA) in ultralytics 8.1.0 is enabled by passing `augment=True` to `model.predict()`. It runs inference at 3 scales (1.0×, 0.83×, 0.67× of imgsz) with horizontal flip, roughly tripling inference time. This task adds an `--augment` flag to evaluate.py for A/B comparison, measures the TTA delta on the weighted score, applies the best configuration to run.py, builds the final submission, and writes the slice verification script.

This task directly delivers requirement **R010** (Test-time augmentation and multi-scale inference to boost detection mAP).

## Steps

1. **Add `--augment` flag to evaluate.py.** Modify the `run_inference` function signature to accept an `augment` parameter (default False). Pass `augment=augment` to `model.predict()`. Add `--augment` to the CLI arg parser as a `store_true` flag. This is a 3-line change:
   - In `run_inference(model, image_dir, conf, imgsz, verbose)` → add `augment: bool = False`
   - In the `model.predict()` call inside `run_inference` → add `augment=augment`
   - In `parse_args()` → add `parser.add_argument("--augment", action="store_true", ...)`
   - In `main()` → pass `args.augment` to `run_inference`

   **Key file locations:**
   - `scripts/evaluate.py` line 80: `def run_inference(model, image_dir, conf, imgsz, verbose)` — add `augment` param
   - `scripts/evaluate.py` line 92: `results = model.predict(source=..., conf=conf, imgsz=imgsz, verbose=False)` — add `augment=augment`
   - `scripts/evaluate.py` line ~171: `parse_args()` function — add `--augment` arg
   - `scripts/evaluate.py` line ~217: `run_inference(model, image_dir, args.conf, args.imgsz, args.verbose)` — add `args.augment`

2. **Run evaluation WITHOUT TTA** (if T01 already recorded this, skip; otherwise run it):
   ```bash
   cd /Users/jama/code/nmai
   venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280
   ```
   Record: det_mAP_base, cls_mAP_base, weighted_base.

3. **Run evaluation WITH TTA:**
   ```bash
   cd /Users/jama/code/nmai
   venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280 --augment
   ```
   Record: det_mAP_tta, cls_mAP_tta, weighted_tta. Note the inference time difference.
   
   **WARNING:** TTA triples inference time. On M1 Max CPU with 50 val images at imgsz=1280, expect ~15-20 min per evaluation (vs ~5-7 min without TTA). This is expected.

4. **Apply the best configuration to run.py.** Compare weighted_base vs weighted_tta.
   - If TTA improves the weighted score: add `augment=True` to the `model.predict()` call in `src/run.py` (line ~131). The call currently reads:
     ```python
     results = model.predict(
         source=str(img_path),
         conf=CONFIDENCE_THRESHOLD,
         imgsz=IMAGE_SIZE,
         device=device,
         verbose=False,
     )
     ```
     Add `augment=True,` after `verbose=False,`.
   - If TTA does NOT improve the weighted score or the delta is negligible: leave run.py unchanged. Add a comment explaining TTA was tested but not beneficial.
   - **Timeout consideration:** On the competition L4 GPU, base inference for 248 images at imgsz=1280 should be ~30-60s. TTA would push to ~90-180s, well within the 300s budget. So timeout is NOT a concern for the decision.

5. **Build final submission and write verify-s04.sh.**
   - Run `venv/bin/python3 scripts/build_submission.py` to build submission.zip with the final config.
   - Write `scripts/verify-s04.sh` with these checks:
     1. models/best.pt exists, >100MB (YOLOv8l), <420MB (submission limit)
     2. Evaluation at imgsz=1280 produces weighted score > 0 (basic sanity)
     3. submission.zip exists and build_submission.py exits 0
     4. src/run.py contains IMAGE_SIZE = 1280
     5. TTA decision is reflected in run.py (augment=True present if TTA helped, absent or commented if not)
     6. models/best.pt is not >420MB (critical submission constraint)

## Must-Haves

- [ ] evaluate.py supports `--augment` flag that enables TTA in inference
- [ ] TTA A/B comparison completed with measured scores for both configurations
- [ ] Best configuration (with or without TTA) applied to src/run.py
- [ ] Final submission.zip built with the best configuration
- [ ] scripts/verify-s04.sh written and passing

## Verification

- `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280 --augment` exits 0
- `bash scripts/verify-s04.sh` passes all checks
- `grep -c 'augment' src/run.py` returns ≥0 (present if TTA helps, may be absent if not)

## Inputs

- `models/best.pt` — stripped YOLOv8l model from T01 (~168MB)
- `src/run.py` — current inference script with IMAGE_SIZE=1280, conf=0.001, no TTA
- `scripts/evaluate.py` — evaluation script (will be modified to add --augment)
- `scripts/build_submission.py` — submission builder (no changes)
- T01 baseline scores: det_mAP, cls_mAP, weighted — from T01 task output

## Expected Output

- `scripts/evaluate.py` — modified with `--augment` flag support
- `src/run.py` — possibly modified with `augment=True` (if TTA improves score)
- `scripts/verify-s04.sh` — new verification script
- `submission.zip` — final submission with best configuration

## Observability Impact

- **New CLI flag:** `scripts/evaluate.py --augment` enables TTA during evaluation inference. Log line `[evaluate] TTA (augment): True/False` confirms which mode ran.
- **TTA decision trail:** `grep -n 'augment' src/run.py` shows whether TTA is baked into the competition inference. Either `augment=True` is present in the predict call, or a `# TTA evaluated but not applied` comment explains why.
- **Verify script:** `bash scripts/verify-s04.sh` is the single-command diagnostic for the entire S04 slice — checks model size, evaluation metrics, submission validity, IMAGE_SIZE, TTA status, and failure-path handling. Exit 0 = all pass.
- **Failure state:** If TTA evaluation crashes, stderr from evaluate.py has the traceback. If verify-s04.sh fails, the specific `❌ FAIL:` line identifies which check broke.
