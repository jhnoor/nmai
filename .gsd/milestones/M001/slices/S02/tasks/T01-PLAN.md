---
estimated_steps: 5
estimated_files: 1
---

# T01: Create sandbox-compliant run.py and smoke-test locally

**Slice:** S02 — Sandbox-Compliant Inference & First Submission
**Milestone:** M001

## Description

Create the competition inference entry point `src/run.py`. This script runs inside a sandboxed environment with strict import restrictions (no os, sys, subprocess, socket, shutil, yaml, etc.) and must produce COCO-format predictions JSON from a directory of shelf images. The sandbox has Python 3.11, PyTorch 2.6.0+cu124, ultralytics 8.1.0, numpy 1.26.4, and an NVIDIA L4 GPU.

The script must include compatibility patches (numpy.trapz shim and torch.load weights_only override) before importing ultralytics, auto-detect CUDA vs CPU, and locate the model weights relative to its own file path (since the zip is extracted flat in the sandbox).

**Relevant skills:** None required — this is a straightforward Python script with known constraints.

## Steps

1. Create the `src/` directory and `src/run.py` with the following structure:
   - Compatibility patches at top of file (before any ultralytics import):
     - `numpy.trapz` shim: `if not hasattr(np, 'trapz'): np.trapz = np.trapezoid`
     - `torch.load` patch: wrap `torch.load` to default `weights_only=False`
   - Import `ultralytics.YOLO` after patches
   - `argparse` CLI accepting `--input` (image directory) and `--output` (predictions JSON path)
   - Model loading: `YOLO(Path(__file__).parent / "best.pt")`
   - Device detection: `"cuda" if torch.cuda.is_available() else "cpu"`
   - Inference loop: iterate all `.jpg`/`.jpeg`/`.png` files in `--input`, run `model.predict()` with `conf=0.001`, `imgsz=640`, `device=device`, `verbose=False`
   - Prediction extraction: for each result, extract `xyxy` boxes, convert to COCO `[x, y, w, h]`, extract `category_id` and `score`
   - `image_id` extraction: from filename `img_XXXXX.jpg` → `int(stem.split('_')[1])` (strips leading zeros)
   - Write predictions list to `--output` as JSON with `json.dump()`
   - Use `Path(args.output).parent.mkdir(parents=True, exist_ok=True)` for defensive dir creation (pathlib, not os)

2. Verify NO blocked imports exist in the file. Check with:
   ```bash
   grep -nE "^import (os|sys|subprocess|socket|ctypes|builtins|importlib|pickle|marshal|shelve|shutil|yaml|requests|urllib|multiprocessing|threading|signal|gc|code|codeop|pty)\b|^from (os|sys|subprocess|socket|ctypes|builtins|importlib|pickle|marshal|shelve|shutil|yaml|requests|urllib|http\.client|multiprocessing|threading|signal|gc|code|codeop|pty) import" src/run.py
   ```
   Must produce zero output.

3. Run the smoke test against training images:
   ```bash
   source venv/bin/activate
   python src/run.py --input norgesgruppen_data/data/train/images --output /tmp/s02_predictions.json
   ```

4. Validate the output format:
   ```python
   import json
   d = json.load(open('/tmp/s02_predictions.json'))
   assert len(d) > 0, "No predictions"
   for p in d[:20]:
       assert set(p.keys()) == {'image_id', 'category_id', 'bbox', 'score'}
       assert isinstance(p['image_id'], int)
       assert isinstance(p['category_id'], int)
       assert isinstance(p['bbox'], list) and len(p['bbox']) == 4
       assert all(isinstance(v, (int, float)) for v in p['bbox'])
       assert isinstance(p['score'], float)
   print(f"OK: {len(d)} predictions, schema valid")
   ```

5. Confirm inference completes in reasonable time. On the full 248 training images, expect ~1–2 minutes on CPU (M1 Max). The sandbox (L4 GPU) will be much faster. Log the elapsed time and prediction count.

## Must-Haves

- [ ] `src/run.py` exists and is executable
- [ ] Zero blocked imports (verified by grep)
- [ ] Includes `numpy.trapz` shim and `torch.load` monkey-patch before ultralytics import
- [ ] Uses `Path(__file__).parent / "best.pt"` for model location
- [ ] Produces valid COCO-format JSON with correct field types
- [ ] `image_id` is int (not string), extracted from `img_XXXXX.jpg` filenames
- [ ] `bbox` is `[x, y, w, h]` in pixels (COCO format, not YOLO normalized)
- [ ] Auto-detects CUDA/CPU device
- [ ] Uses `conf=0.001` for maximum recall

## Verification

- `grep -cE "^import (os|sys|subprocess|socket|shutil|yaml)" src/run.py` returns `0`
- `python src/run.py --input norgesgruppen_data/data/train/images --output /tmp/s02_predictions.json` exits 0
- `/tmp/s02_predictions.json` contains >1000 predictions (248 images × ~50 avg = ~14k expected)
- All predictions match the schema `{image_id: int, category_id: int, bbox: [4 floats], score: float}`

## Inputs

- `models/best.pt` — S01 trained YOLOv8m checkpoint (52MB, nc=356, detection mAP@0.5=0.671)
- `norgesgruppen_data/data/train/images/` — 248 shelf images for smoke testing
- `scripts/train.py` lines 1–30 — canonical monkey-patch pattern for numpy.trapz and torch.load (copy this pattern)

### Key knowledge from S01:
- Every script that imports ultralytics must apply numpy.trapz and torch.load patches BEFORE the import
- The torch.load patch uses `functools.wraps` — `functools` is safe (not blocked)
- `conf=0.001` produces ~14k predictions on 50 val images — expect ~55k on all 248 images
- Model was trained with nc=356 (IDs 0–355). Category 356 (unknown_product) won't appear in predictions.
- Sandbox is Python 3.11 — avoid 3.12-only syntax like `type` keyword or `ExceptionGroup`

## Expected Output

- `src/run.py` — sandbox-compliant inference script (~80–120 lines)
- `/tmp/s02_predictions.json` — valid predictions file (artifact, not committed)
