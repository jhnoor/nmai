---
estimated_steps: 5
estimated_files: 3
---

# T01: Add training tuning args and launch improved training run

**Slice:** S03 — Detection Tuning & Classification
**Milestone:** M001

## Description

Add CLI arguments for `close_mosaic` and `cls` (classification loss weight) to `scripts/train.py`, update `src/run.py` to use IMAGE_SIZE=1280, write the S03 verification script, and launch an improved training run with YOLOv8l at imgsz=1280.

The research (S03-RESEARCH.md) identified three high-impact levers: (1) imgsz=1280 to rescue small objects lost at 640px, (2) close_mosaic=0 to prevent the mAP cliff observed at epoch 71 in S01, (3) cls=1.0 to push classification harder on 356 classes. YOLOv8l adds model capacity.

**Important context:**
- Training takes 8-12 hours on CPU. Start it in the background and let T02 harvest results.
- CPU only — MPS is broken (D008/D009). Use `--device cpu`.
- If YOLOv8l OOMs at batch=2 imgsz=1280, fall back to YOLOv8m with same settings.
- The venv is at `venv/` — all Python commands use `venv/bin/python3`.
- ultralytics 8.1.0 requires numpy/torch patches BEFORE import — these already exist in train.py.
- ultralytics saves training outputs to its own project root, NOT CWD. train.py already handles this via `model.trainer.save_dir`.
- The project root is `/Users/jama/code/nmai`. All commands run from there.

## Steps

1. **Add `--close-mosaic` and `--cls` CLI args to `scripts/train.py`:**
   - Add `--close-mosaic` as `type=int, default=10` (ultralytics default is 10; we want to pass 0).
   - Add `--cls` as `type=float, default=0.5` (ultralytics default is 0.5; we want 1.0).
   - Pass `close_mosaic=args.close_mosaic` to the `model.train()` call (add to the function call kwargs, NOT inside the `augmentation` dict).
   - Pass `cls=args.cls` to the `model.train()` call similarly.
   - Also pass these to the MPS-fallback `model.train()` call.
   - Print the new arg values in the header log lines.

2. **Update `src/run.py` IMAGE_SIZE from 640 to 1280:**
   - Change the line `IMAGE_SIZE = 640` to `IMAGE_SIZE = 1280`.
   - This matches the training resolution. ultralytics auto-resizes but explicit matching is better.

3. **Write `scripts/verify-s03.sh`:**
   - Make it executable (`chmod +x`).
   - It should check:
     - (a) `models/best.pt` exists and is >1MB (new model present)
     - (b) Run evaluation: `venv/bin/python3 scripts/evaluate.py --model models/best.pt --data data/yolo_dataset/dataset.yaml --imgsz 1280` and capture output
     - (c) Parse weighted score from evaluation output, check > 0.578 (S02 baseline)
     - (d) Parse detection mAP, check ≥ 0.50 (sanity — shouldn't regress catastrophically)
     - (e) Check `src/run.py` contains `IMAGE_SIZE = 1280`
     - (f) Run `venv/bin/python3 scripts/build_submission.py` and check exit 0
     - (g) Check `submission.zip` exists after rebuild
   - Use same structure as verify-s02.sh: PASS/FAIL counters, summary at end.
   - Use `source venv/bin/activate` at the top.

4. **Launch the training run:**
   - Command: `cd /Users/jama/code/nmai && venv/bin/python3 scripts/train.py --model yolov8l.pt --imgsz 1280 --batch 2 --epochs 80 --device cpu --close-mosaic 0 --cls 1.0 --name s03_yolov8l_1280`
   - Start this in the background (use bg_shell or nohup). Training will take 8-12 hours.
   - If the first few epochs show OOM or crash, fall back to: `--model yolov8m.pt --batch 4` (keep imgsz=1280, close_mosaic=0, cls=1.0).
   - Monitor the first 1-2 epochs to confirm training is progressing before declaring T01 done.

5. **Verify T01 deliverables:**
   - `venv/bin/python3 scripts/train.py --help` shows `--close-mosaic` and `--cls` args
   - `grep "IMAGE_SIZE" src/run.py` shows 1280
   - `test -x scripts/verify-s03.sh` confirms script is executable
   - Training process is running (check with ps or bg_shell status)

## Must-Haves

- [ ] `scripts/train.py` has `--close-mosaic` (int) and `--cls` (float) CLI args passed to model.train()
- [ ] `src/run.py` IMAGE_SIZE changed from 640 to 1280
- [ ] `scripts/verify-s03.sh` exists, is executable, checks model/eval/submission
- [ ] Training process launched and running (first epoch started without OOM)

## Verification

- `venv/bin/python3 scripts/train.py --help` shows --close-mosaic and --cls in output
- `grep 'IMAGE_SIZE = 1280' src/run.py` matches
- `test -x scripts/verify-s03.sh && echo OK` prints OK
- Training process is alive (bg_shell digest or ps shows python training)

## Inputs

- `scripts/train.py` — existing training script, needs --close-mosaic and --cls args added
- `src/run.py` — existing inference script, needs IMAGE_SIZE updated to 1280
- `scripts/verify-s02.sh` — reference for verify script structure/style
- S03-RESEARCH.md findings: YOLOv8l, imgsz=1280, close_mosaic=0, cls=1.0, batch=2, epochs=80

## Expected Output

- `scripts/train.py` — updated with two new CLI args, both passed through to model.train()
- `src/run.py` — IMAGE_SIZE = 1280 (was 640)
- `scripts/verify-s03.sh` — new executable verification script (~80-120 lines)
- Background training process running YOLOv8l at imgsz=1280

## Observability Impact

- **New signals:** `train.py` prints `close_mosaic` and `cls` values in header log, making training config inspectable. `verify-s03.sh` checks train.py --help output to confirm args are wired.
- **Inspection:** `venv/bin/python3 scripts/train.py --help` shows new args. `bg_shell digest` or `bg_shell output` shows training progress (epoch, loss values). `grep IMAGE_SIZE src/run.py` confirms inference resolution.
- **Failure visibility:** If training OOMs — bg_shell status shows crashed process, stdout contains error. If training diverges — epoch loss values increase instead of decreasing (visible in bg_shell output). If args not passed — model.train() uses ultralytics defaults (close_mosaic=10, cls=0.5) silently, but verify-s03.sh Step 8 catches missing args in --help.
