---
estimated_steps: 5
estimated_files: 3
---

# T01: Complete S03 training handoff — strip optimizer, evaluate, and build baseline submission

**Slice:** S04 — Score Maximization & Final Submissions
**Milestone:** M001

## Description

S03 launched a YOLOv8l training run (imgsz=1280, batch=2, epochs=80, close_mosaic=0, cls=1.0) as PID 9132. That process is running with the OLD train.py code (before the optimizer-stripping fix), so manual stripping is mandatory. This task checks training status, strips the optimizer from the best checkpoint, copies to models/best.pt, evaluates the model, and builds a baseline submission.zip.

**Critical context:** The running training process uses old code that does NOT strip optimizer state. The raw checkpoint is ~503MB (over the 420MB limit). You MUST strip the optimizer manually.

## Steps

1. **Check training completion.** Inspect `runs/detect/s03_yolov8l_1280/results.csv` — it should have 80 data rows (plus 1 header). If the training process (PID 9132) is still alive (`ps aux | grep 'train.py.*s03_yolov8l' | grep -v grep`), training hasn't finished yet. If training has NOT completed but a `best.pt` checkpoint exists, you can proceed with whatever checkpoint is available — note the epoch count in your output. If training IS still running and is very early (< epoch 20), the checkpoint quality may be too low to be useful — in that case, note the limitation and proceed anyway (S04 depends on whatever model is available).

2. **Strip optimizer from the best checkpoint.** The checkpoint at `runs/detect/s03_yolov8l_1280/weights/best.pt` is ~503MB with optimizer state. Strip it:
   ```bash
   cd /Users/jama/code/nmai
   venv/bin/python3 -c "
   import functools, torch, numpy as np
   if not hasattr(np, 'trapz'): np.trapz = np.trapezoid
   _orig = torch.load
   @functools.wraps(_orig)
   def _patched(*args, **kwargs):
       if 'weights_only' not in kwargs: kwargs['weights_only'] = False
       return _orig(*args, **kwargs)
   torch.load = _patched
   ckpt = torch.load('runs/detect/s03_yolov8l_1280/weights/best.pt')
   print(f'Keys: {list(ckpt.keys())}')
   print(f'Epoch: {ckpt.get(\"epoch\", \"unknown\")}')
   ckpt['optimizer'] = None
   torch.save(ckpt, 'models/best.pt')
   import os; print(f'Stripped: {os.path.getsize(\"models/best.pt\") / 1024 / 1024:.1f} MB')
   "
   ```
   Verify the file is >100MB (real YOLOv8l) and <420MB (submission limit).

3. **Evaluate the model at imgsz=1280.** Run:
   ```bash
   cd /Users/jama/code/nmai
   venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280
   ```
   Record the output: detection mAP, classification mAP, and weighted score. This is the **baseline without TTA** that T02 will compare against.

4. **Build submission.zip.** Run:
   ```bash
   cd /Users/jama/code/nmai
   venv/bin/python3 scripts/build_submission.py
   ```
   Confirm it exits 0 and submission.zip is created.

5. **Record results.** Print a clear summary of: training epoch count, model file size, detection mAP, classification mAP, weighted score. This output is consumed by T02 for TTA comparison.

## Must-Haves

- [ ] models/best.pt is the stripped YOLOv8l checkpoint (>100MB, <420MB)
- [ ] Evaluation runs successfully at imgsz=1280 and produces weighted score
- [ ] submission.zip built and validated by build_submission.py
- [ ] Baseline scores (det_mAP, cls_mAP, weighted) are recorded in task output

## Verification

- `ls -la models/best.pt` — file size >100MB and <420MB
- `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280` exits 0 with structured metric output
- `venv/bin/python3 scripts/build_submission.py` exits 0
- `ls -la submission.zip` — file exists

## Inputs

- `runs/detect/s03_yolov8l_1280/weights/best.pt` — S03 training output checkpoint (~503MB with optimizer)
- `scripts/evaluate.py` — evaluation script (no changes needed)
- `scripts/build_submission.py` — submission builder (no changes needed)
- `data/yolo_dataset/dataset.yaml` — dataset config for evaluation
- S03 summary: training launched as PID 9132, using OLD train.py without optimizer stripping fix. Manual stripping is mandatory. The numpy and torch compatibility patches are required (see step 2).

## Expected Output

- `models/best.pt` — stripped YOLOv8l model (~168MB)
- `submission.zip` — built with the YOLOv8l model
- Baseline evaluation scores recorded in task output for T02 consumption

## Observability Impact

- **New signal:** `models/best.pt` file size is the primary health indicator — >100MB confirms real YOLOv8l weights, <420MB confirms optimizer stripped.
- **Inspection surface:** `venv/bin/python3 -c "import torch; ckpt=torch.load('models/best.pt',weights_only=False); print('epoch:', ckpt.get('epoch'), 'optimizer:', ckpt.get('optimizer'))"` — should show the training epoch and `optimizer: None`.
- **Failure state:** If stripping fails, models/best.pt will be >420MB or missing. If evaluation fails, no weighted score line in stdout and exit code != 0. If submission build fails, no submission.zip created.
- **Future agent:** T02 reads the baseline scores from T01's summary to compare with TTA results. The evaluation command output format (structured metric lines) is the contract.
