# S02: Sandbox-Compliant Inference & First Submission

**Goal:** A working `src/run.py` that passes the sandbox security scan and produces correct COCO-format predictions, plus a `scripts/build_submission.py` that creates a valid submission zip ready for upload.
**Demo:** `python src/run.py --input norgesgruppen_data/data/train/images --output /tmp/predictions.json` produces valid predictions JSON; `python scripts/build_submission.py` creates `submission.zip` passing all competition constraints; no blocked imports in `src/run.py`.

## Must-Haves

- `src/run.py` uses only allowed imports (pathlib, json, argparse, functools, torch, ultralytics — NO os/sys/subprocess/etc.)
- `src/run.py` includes torch.load monkey-patch for PyTorch 2.6.0 compatibility (required for sandbox)
- `src/run.py` writes COCO-format JSON: `[{image_id: int, category_id: int, bbox: [x,y,w,h], score: float}]`
- `src/run.py` locates `best.pt` relative to `Path(__file__).parent` (zip extracts flat)
- `src/run.py` auto-detects CUDA/CPU device (sandbox has NVIDIA L4)
- `scripts/build_submission.py` creates `submission.zip` with `run.py` and `best.pt` at root
- Zip respects all limits: ≤420MB uncompressed, ≤3 weight files, ≤10 Python files, allowed file types only
- Predictions smoke-test passes on local training images with correct field types and schema

## Proof Level

- This slice proves: integration (sandbox-compatible inference + valid submission structure)
- Real runtime required: yes (model inference must actually run)
- Human/UAT required: yes (leaderboard submission requires manual upload at https://app.ainm.no/submit/norgesgruppen-data)

## Verification

- `bash scripts/verify-s02.sh` — end-to-end slice verification covering:
  - No blocked imports in `src/run.py` (grep-based security scan simulation)
  - `src/run.py` produces valid predictions JSON on a subset of training images
  - Predictions have correct schema (image_id:int, category_id:int, bbox:list, score:float)
  - `scripts/build_submission.py` produces `submission.zip`
  - Zip contains `run.py` and `best.pt` at root (no subdirectories)
  - Zip size is under 420MB uncompressed
  - No disallowed file types in zip

## Integration Closure

- Upstream surfaces consumed: `models/best.pt` (S01 trained weights), `scripts/evaluate.py` (S01 local eval for pre-submission validation)
- New wiring introduced: `src/run.py` (competition entry point), `scripts/build_submission.py` (zip builder)
- What remains before the milestone is truly usable end-to-end: S03 (classification tuning), S04 (score maximization), and manual upload of `submission.zip` to the leaderboard

## Observability / Diagnostics

- **Runtime signals:** `src/run.py` prints a progress line per image (`[N/M] img_XXXXX.jpg — K detections`) to stderr so inference progress is visible. Final summary line reports total predictions, elapsed time, and output path.
- **Inspection surfaces:** The predictions JSON at `--output` is the primary inspection artifact. `python -c "import json; d=json.load(open('/tmp/predictions.json')); print(len(d))"` gives quick count. `jq '.[0]' /tmp/predictions.json` shows schema of first prediction.
- **Failure visibility:** If the model file is missing, run.py exits with code 1 and a clear error message naming the expected path. If no images are found in `--input`, it writes an empty JSON array and prints a warning. If an individual image fails inference, the error is printed to stderr and that image is skipped (not fatal).
- **Submission zip diagnostics:** `scripts/build_submission.py` prints a validation summary: file count, total uncompressed size, and pass/fail for each constraint. `unzip -l submission.zip` shows contents.
- **Redaction:** No secrets or credentials in any artifact. Model weights are binary; predictions JSON contains only numeric fields.

## Verification (Failure-Path Check)

In addition to the happy-path checks above, `scripts/verify-s02.sh` includes:
- A check that `src/run.py` exits non-zero when given a non-existent input directory
- A check that `src/run.py` produces an empty predictions array `[]` when given a directory with no images

## Tasks

- [x] **T01: Create sandbox-compliant run.py and smoke-test locally** `est:45m`
  - Why: This is the core submission entry point — must avoid all blocked imports, include the torch.load compatibility patch, produce correct COCO-format predictions, and auto-detect device. Highest-risk artifact in the slice.
  - Files: `src/run.py`
  - Do: Create `src/run.py` with: (1) numpy.trapz shim + torch.load monkey-patch before ultralytics import, (2) argparse CLI with `--input` and `--output`, (3) model loading from `Path(__file__).parent / "best.pt"`, (4) CUDA/CPU auto-detection, (5) inference loop over all .jpg/.jpeg/.png images in input dir, (6) COCO-format output with `image_id` as int from filename, `category_id` from model, `bbox` as [x,y,w,h] in pixels, `score` as float. Use `conf=0.001` for maximum recall. Use `pathlib` for all file ops. Ensure `Path(output).parent.mkdir(parents=True, exist_ok=True)` for defensive output dir creation.
  - Verify: `grep -E "^import (os|sys|subprocess|socket|shutil|yaml)" src/run.py` returns nothing; `source venv/bin/activate && python src/run.py --input norgesgruppen_data/data/train/images --output /tmp/predictions.json && python -c "import json; d=json.load(open('/tmp/predictions.json')); print(f'{len(d)} predictions'); assert len(d)>0; assert all(isinstance(p['image_id'],int) and isinstance(p['category_id'],int) and isinstance(p['bbox'],list) and len(p['bbox'])==4 and isinstance(p['score'],float) for p in d)"`
  - Done when: `src/run.py` exists, contains zero blocked imports, and produces a valid predictions JSON with >0 predictions from training images.

- [x] **T02: Create build_submission.py, verification script, and validate the complete zip** `est:30m`
  - Why: The submission zip must meet strict competition constraints (file types, sizes, structure). A builder script with validation catches errors before wasting one of the 3 daily submission attempts. The verification script provides one-command proof the slice is done.
  - Files: `scripts/build_submission.py`, `scripts/verify-s02.sh`
  - Do: (1) Create `scripts/build_submission.py` that: copies `src/run.py` → staging as `run.py` and `models/best.pt` → staging as `best.pt`, builds `submission.zip` using `zipfile` module, validates zip contents (run.py at root, ≤3 weight files, ≤10 .py files, total uncompressed ≤420MB, only allowed extensions). (2) Create `scripts/verify-s02.sh` that runs the blocked-import check, the smoke-test, and the zip builder with output validation. (3) Run the verification script end-to-end.
  - Verify: `bash scripts/verify-s02.sh` exits 0; `unzip -l submission.zip` shows `run.py` and `best.pt` at root with no subdirectories.
  - Done when: `submission.zip` exists at repo root, passes all validation checks, and `scripts/verify-s02.sh` exits 0.

## Files Likely Touched

- `src/run.py` (create)
- `scripts/build_submission.py` (create)
- `scripts/verify-s02.sh` (create)
- `submission.zip` (generated artifact)
