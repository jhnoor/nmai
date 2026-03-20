# S02: Sandbox-Compliant Inference & First Submission — Research

**Date:** 2026-03-20
**Depth:** Light research — straightforward work with known patterns, clear contract from README, no ambiguous APIs.

## Summary

S02 creates two artifacts: `src/run.py` (sandbox inference script) and `scripts/build_submission.py` (zip builder with validation). The work is well-defined by the competition README's submission contract and the sandbox constraints. The model (`models/best.pt`, 50MB, nc=356) is ready from S01. The key risk is ultralytics 8.1.0 compatibility in the sandbox — specifically the `torch.load weights_only` patch needed for PyTorch 2.6.0. The numpy.trapz patch is NOT needed in the sandbox (it ships numpy 1.26.4 which still has `np.trapz`), but is harmless to include.

All blocked imports are avoidable: `run.py` needs only `pathlib`, `json`, `argparse`, `torch`, `functools`, and `ultralytics`. The security scan only checks user-submitted `.py` files — ultralytics itself uses `os`/`sys`/`yaml` internally, which is fine since it's pre-installed.

## Recommendation

Build a single self-contained `run.py` with no helper files. Include the `torch.load` compatibility patch inline (needed for PyTorch 2.6.0 sandbox). Use `conf=0.001` for maximum recall (mAP evaluation benefits from all predictions ranked by confidence). Use `Path(__file__).parent / "best.pt"` to locate the model relative to the script. Build the zip with `zipfile` module in `build_submission.py`, including validation checks for all competition constraints.

## Implementation Landscape

### Key Files

- `src/run.py` — **CREATE.** Sandbox inference entry point. Accepts `--input` (image dir) and `--output` (predictions JSON path). Loads model from same directory, runs YOLO inference on all images, writes COCO-format predictions JSON. Must avoid all blocked imports (os, sys, subprocess, socket, yaml, shutil, etc.). Uses pathlib for all file operations.
- `scripts/build_submission.py` — **CREATE.** Builds submission.zip by copying `src/run.py` and `models/best.pt` to a staging directory, then zipping. Validates: run.py at root, weight count ≤3, Python file count ≤10, total uncompressed size ≤420MB, no blocked file types.
- `models/best.pt` — **EXISTS.** 50MB YOLOv8m checkpoint, nc=356 (IDs 0-355). No changes needed.
- `scripts/evaluate.py` — **EXISTS.** Used for pre-submission local validation to confirm the model produces reasonable scores before spending a submission attempt.

### Build Order

1. **`src/run.py` first** — this is the core deliverable and the riskiest piece (blocked-import compliance, correct output format, device detection). Can be smoke-tested locally immediately against training images.
2. **`scripts/build_submission.py` second** — mechanical zip builder. Depends on run.py existing. Include structure validation to catch errors before wasting a submission.
3. **Local smoke test** — run `python src/run.py --input norgesgruppen_data/data/train/images --output /tmp/predictions.json` and validate output format against the COCO schema.
4. **Pre-submission eval** — run evaluate.py on the predictions to verify scores are reasonable.
5. **Build and submit** — create zip, upload to leaderboard, get first score.

### Verification Approach

**Smoke test for run.py:**
```bash
python src/run.py --input norgesgruppen_data/data/train/images --output /tmp/predictions.json
python -c "import json; d=json.load(open('/tmp/predictions.json')); print(f'{len(d)} predictions'); assert all(set(p.keys()) == {'image_id','category_id','bbox','score'} for p in d[:10])"
```

**Security scan simulation:**
```bash
# Verify no blocked imports in run.py
grep -E "^import (os|sys|subprocess|socket|ctypes|builtins|importlib|pickle|marshal|shelve|shutil|yaml|requests|urllib|multiprocessing|threading|signal|gc|code|codeop|pty)\b|^from (os|sys|subprocess|socket|ctypes|builtins|importlib|pickle|marshal|shelve|shutil|yaml|requests|urllib|http\.client|multiprocessing|threading|signal|gc|code|codeop|pty) import" src/run.py
# Should produce zero output
```

**Zip structure validation:**
```bash
python scripts/build_submission.py
unzip -l submission.zip  # Should show run.py and best.pt at root
```

**Pre-submission local eval:**
```bash
python scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml
# Expect: det_mAP ≈ 0.671, weighted ≈ 0.578
```

## Constraints

- **Blocked imports (hard):** os, sys, subprocess, socket, ctypes, builtins, importlib, pickle, marshal, shelve, shutil, yaml, requests, urllib, http.client, multiprocessing, threading, signal, gc, code, codeop, pty. Also blocked calls: eval(), exec(), compile(), `__import__()`.
- **Sandbox numpy is 1.26.4** — has `np.trapz` natively, so the trapz patch is unnecessary but harmless.
- **Sandbox PyTorch is 2.6.0** — defaults `weights_only=True` in `torch.load`, so the `functools.wraps` patch IS required for model loading.
- **Model path:** In sandbox, all zip contents are extracted to one directory. `run.py` must locate `best.pt` relative to `Path(__file__).parent`.
- **Output contract:** JSON array of `{image_id: int, category_id: int, bbox: [x,y,w,h], score: float}`. `image_id` extracted from filename `img_XXXXX.jpg → XXXXX` (as integer, leading zeros stripped).
- **Device:** Sandbox has NVIDIA L4 GPU. Use `torch.cuda.is_available()` to auto-detect. Must work on CPU too (for local testing).
- **Timeout: 300s.** Estimated ~0.36s/image on CPU (M1 Max), much faster on L4 GPU. Even 500 images at 0.36s = 180s — well within budget.
- **Max files:** ≤10 Python, ≤3 weight files, ≤420MB uncompressed. Our submission: 1 Python file + 1 weight file (50MB) — well within limits.

## Common Pitfalls

- **`argparse` uses `sys` internally** — argparse imports sys under the hood. If the security scan does AST-level import checking on submitted files only (which it does per the README), this is fine. But if it does runtime import tracing, this would fail. Evidence from the README: "Security scan found violations" → "Remove imports of subprocess, socket, os, etc." — phrased as user code imports, not transitive. **Mitigation:** Use argparse normally; if submission fails, fall back to manual argv parsing.
- **Forgetting `functools` import for torch.load patch** — the patch requires `import functools`. This IS safe (not blocked).
- **Writing predictions to wrong path** — The `--output` argument is the full file path (e.g., `/output/predictions.json`), not a directory. Write directly to it. The parent directory should already exist in the sandbox.
- **`image_id` as string vs int** — The contract requires `int`. Extract from `img_00042.jpg → 42` (not `"00042"`). Use `int(stem.split('_')[1])` pattern.
- **Missing parent directory for output** — In sandbox, `/output/` may not exist. Use `Path(output).parent.mkdir(parents=True, exist_ok=True)` as defensive code. `mkdir` is a pathlib method, not os — this is safe.
- **`torch.cuda.is_available()` import path** — `torch` is allowed. Don't accidentally import `torch.cuda` separately (not needed — `torch.cuda.is_available()` works from the main torch import).
