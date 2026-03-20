---
estimated_steps: 4
estimated_files: 3
---

# T02: Build training script, benchmark MPS vs CPU, and run full training

**Slice:** S01 — Data Pipeline & Detection Baseline
**Milestone:** M001

## Description

Write the YOLOv8m fine-tuning script, benchmark MPS vs CPU training speed on this dataset (5 epochs each), then run a full training with the faster backend. This task produces `models/best.pt` — the trained detection model that S02 needs for inference. The benchmark retires the MPS speed risk identified in the roadmap.

**Skills to load if available:** none needed — ultralytics handles the training loop.

## Steps

1. **Create `scripts/` dir if needed and write `scripts/train.py`.** The script must:
   - Accept CLI args: `--data` (path to dataset.yaml, default: `data/yolo_dataset/dataset.yaml`), `--model` (pretrained model, default: `yolov8m.pt`), `--device` (mps/cpu/0, default: auto-detect), `--epochs` (default: 80), `--imgsz` (default: 640), `--batch` (default: 16), `--name` (run name, default: `train`)
   - Load the pretrained model: `from ultralytics import YOLO; model = YOLO(args.model)`
   - Call `model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, name=args.name, ...)`
   - Include sensible augmentation defaults for a small dataset (248 images): `mosaic=1.0, mixup=0.15, copy_paste=0.0, flipud=0.5, fliplr=0.5, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`
   - After training completes, print the path to `best.pt` and copy it to `models/best.pt` using shutil
   - Use `yolov8m.pt` as model name — this is the correct name for ultralytics 8.1.0 (NOT `yolo26m.pt` which is a newer version naming)
   
   **Important constraints:**
   - ultralytics 8.1.0 API: `YOLO("yolov8m.pt")` downloads pretrained weights on first use. This is fine since we're on a network-connected dev machine.
   - The `model.train()` call returns results with metrics. Print the final metrics.
   - If MPS causes issues (OOM, crashes), fall back to CPU with batch=8.
   - Set `patience=20` for early stopping to prevent overfitting on this small dataset.
   - Set `save=True, save_period=-1` to save only best and last checkpoints.

2. **Run MPS vs CPU benchmark.** Execute 5-epoch training runs on both backends:
   ```bash
   source venv/bin/activate
   # MPS benchmark
   time python scripts/train.py --device mps --epochs 5 --name bench_mps --batch 16
   # CPU benchmark
   time python scripts/train.py --device cpu --epochs 5 --name bench_cpu --batch 8
   ```
   Record wall time for each. Document which is faster and by how much. If MPS fails (OOM, crash, NaN loss), document the error and use CPU.

3. **Run full training with the faster backend.** Using the winner from the benchmark:
   ```bash
   python scripts/train.py --device <winner> --epochs 80 --name full_train --batch <appropriate>
   ```
   This will take a while (likely 30–90 minutes depending on backend). Monitor for:
   - Loss decreasing over epochs
   - mAP metrics improving
   - No NaN losses or crashes
   - Early stopping may trigger before 80 epochs if val loss plateaus

4. **Copy best weights and verify.** After training completes:
   - The script should auto-copy `best.pt` to `models/best.pt`
   - Verify: `ls -la models/best.pt` shows file > 1MB
   - Check training output: `cat runs/detect/full_train/results.csv | tail -5` shows metrics
   - Document: final training mAP, number of epochs completed, device used, wall time

## Must-Haves

- [ ] `scripts/train.py` with configurable device, epochs, imgsz, batch, augmentation params
- [ ] MPS vs CPU benchmark completed (5 epochs each, wall time recorded)
- [ ] Full training run completed (50+ epochs or early-stopped)
- [ ] `models/best.pt` exists and is a valid YOLOv8m checkpoint > 1MB
- [ ] Training metrics (loss, mAP) documented

## Verification

- `python scripts/train.py --help` shows all expected arguments
- `ls -la models/best.pt` shows file exists and is > 40MB (YOLOv8m is ~50MB)
- `python -c "from ultralytics import YOLO; m=YOLO('models/best.pt'); print(m.info())"` loads without error
- Training run produced `runs/detect/full_train/` with `results.csv` and `weights/best.pt`

## Inputs

- `data/yolo_dataset/dataset.yaml` — from T01, YOLO dataset config with nc=356
- `data/yolo_dataset/images/train/` — ~198 training images (symlinks)
- `data/yolo_dataset/labels/train/` — ~198 training label files
- `venv/` — Python 3.12 venv with ultralytics==8.1.0 from T01

## Expected Output

- `scripts/train.py` — YOLOv8m fine-tuning script (~60–80 lines)
- `models/best.pt` — trained YOLOv8m weights (best checkpoint from full training)
- `runs/detect/` — ultralytics training output directories (bench_mps, bench_cpu, full_train) with metrics and logs
- Benchmark result documented: which device is faster and by how much

## Observability Impact

- **Training logs**: ultralytics writes per-epoch metrics (box_loss, cls_loss, dfl_loss, precision, recall, mAP50, mAP50-95) to `runs/detect/<name>/results.csv`. A future agent can inspect training curves by reading this CSV.
- **Device selection**: train.py prints `[train] Device selected: <device>` at startup. If MPS fails, it logs `[train] MPS training failed: <error>` and `[train] Falling back to CPU with batch=8...`.
- **Model output**: train.py prints the final model path, copy destination, and model size on completion. A future agent can verify by checking `models/best.pt` existence and size.
- **Failure visibility**: MPS crashes are caught and logged with the exception message before falling back. Training failures on the fallback device propagate as unhandled exceptions with full tracebacks.
