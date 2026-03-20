---
estimated_steps: 5
estimated_files: 3
---

# T02: Create build_submission.py, verification script, and validate the complete zip

**Slice:** S02 — Sandbox-Compliant Inference & First Submission
**Milestone:** M001

## Description

Create the submission zip builder (`scripts/build_submission.py`) and the slice verification script (`scripts/verify-s02.sh`). The builder copies `src/run.py` and `models/best.pt` into a flat zip at the repo root, then validates all competition constraints. The verification script runs the full chain: blocked-import check → smoke-test inference → zip build → zip validation.

## Steps

1. Create `scripts/build_submission.py` that:
   - Creates a temporary staging directory (use `tempfile.mkdtemp()`)
   - Copies `src/run.py` → staging as `run.py` (at root, no subdirectory)
   - Copies `models/best.pt` → staging as `best.pt` (at root)
   - Builds `submission.zip` at repo root using `zipfile.ZipFile` (write mode)
   - Adds files with `arcname` set to just the filename (ensures flat structure)
   - Runs validation checks on the built zip:
     - `run.py` exists at root
     - Weight file count ≤3
     - Python file count ≤10
     - Total uncompressed size ≤420MB (440,401,920 bytes)
     - All files have allowed extensions (`.py`, `.json`, `.yaml`, `.yml`, `.cfg`, `.pt`, `.pth`, `.onnx`, `.safetensors`, `.npy`)
   - Prints summary: file count, total size, weight count, Python file count
   - Exits with code 0 on success, 1 on validation failure
   - Clean up staging directory after zip creation

2. Create `scripts/verify-s02.sh` (executable) that runs the full slice verification:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   
   # Step 1: Check for blocked imports in src/run.py
   # Step 2: Run inference smoke test on a SUBSET of images (first 10)
   #   - Create a temp dir, symlink 10 images, run run.py, validate output JSON
   # Step 3: Build submission zip
   # Step 4: Validate zip contents (run.py and best.pt at root, no subdirs)
   # Step 5: Validate zip size
   ```
   
   Use a subset of images (first 10) for the smoke test to keep verification fast (<30s on CPU). The full 248-image test is done in T01; this is a quick sanity check.

3. Run `bash scripts/verify-s02.sh` and confirm exit 0.

4. Verify `submission.zip` exists and inspect contents with `unzip -l submission.zip`.

5. Run `python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml` as a pre-submission sanity check to confirm the model still produces the expected scores (det_mAP ≈ 0.671, weighted ≈ 0.578). This doesn't test run.py specifically but confirms the weights are good.

## Must-Haves

- [ ] `scripts/build_submission.py` creates `submission.zip` at repo root
- [ ] Zip contains `run.py` and `best.pt` at root (no subdirectories — verified by arcname)
- [ ] Zip passes all size and count limits from competition README
- [ ] Only allowed file extensions in zip
- [ ] `scripts/verify-s02.sh` runs blocked-import check + smoke test + zip build + zip validation
- [ ] `scripts/verify-s02.sh` exits 0

## Verification

- `python scripts/build_submission.py` exits 0 and prints validation summary
- `unzip -l submission.zip` shows exactly 2 entries: `run.py` and `best.pt`, no directory prefixes
- `bash scripts/verify-s02.sh` exits 0
- `submission.zip` uncompressed size is <420MB (expect ~52MB: 50MB weights + 2KB script)

## Inputs

- `src/run.py` — sandbox-compliant inference script from T01
- `models/best.pt` — S01 trained weights (52MB)
- `scripts/evaluate.py` — S01 evaluation script (for pre-submission sanity check, not modified)
- `data/yolo_dataset/dataset.yaml` — S01 dataset config (for evaluate.py)
- `norgesgruppen_data/data/train/images/` — training images for smoke test subset

### Competition constraints (from README):
- Max zip uncompressed: 420 MB
- Max Python files: 10
- Max weight files (.pt, .pth, .onnx, .safetensors, .npy): 3
- Max total files: 1000
- Allowed types: .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy
- run.py MUST be at zip root (not in a subfolder)

## Expected Output

- `scripts/build_submission.py` — zip builder with validation (~80–120 lines)
- `scripts/verify-s02.sh` — end-to-end S02 verification script (~60–80 lines)
- `submission.zip` — ready-to-upload competition submission (~50MB)
