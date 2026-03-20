#!/usr/bin/env python3
"""YOLOv8m fine-tuning script for NorgesGruppen shelf dataset.

Supports configurable device (mps/cpu/cuda), epochs, image size, batch size,
and augmentation parameters. After training, copies best.pt to models/best.pt.
"""

import argparse
import functools
import shutil
import time
from pathlib import Path

import numpy as np
import torch

# Patch numpy.trapz for numpy 2.0+ compatibility with ultralytics 8.1.0
# np.trapz was deprecated in numpy 2.0 and removed in 2.4+; replaced by np.trapezoid
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# Patch torch.load to use weights_only=False for ultralytics 8.1.0 compatibility
# with PyTorch 2.6+ (which defaults to weights_only=True).
# Safe because we only load official ultralytics pretrained weights.
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO  # noqa: E402 — must import after patch


def detect_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8m on NorgesGruppen shelf dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/yolo_dataset/dataset.yaml",
        help="Path to dataset.yaml (default: data/yolo_dataset/dataset.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Pretrained model name or path (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device: mps, cpu, or 0 for CUDA (default: auto-detect)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs (default: 80)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name for output directory (default: train)",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Epoch to disable mosaic augmentation (0=never disable, default: 10)",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=0.5,
        help="Classification loss weight (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    device = args.device if args.device else detect_device()
    print(f"[train] Device selected: {device}")
    print(f"[train] Model: {args.model}")
    print(f"[train] Dataset: {args.data}")
    print(f"[train] Epochs: {args.epochs}, ImgSz: {args.imgsz}, Batch: {args.batch}")
    print(f"[train] Close mosaic: {args.close_mosaic}, Cls loss weight: {args.cls}")
    print(f"[train] Run name: {args.name}")

    # Load pretrained model
    model = YOLO(args.model)

    # Augmentation defaults tuned for small dataset (~248 images)
    augmentation = dict(
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    start_time = time.time()

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            name=args.name,
            patience=20,
            save=True,
            save_period=-1,
            close_mosaic=args.close_mosaic,
            cls=args.cls,
            **augmentation,
        )
    except Exception as e:
        # If MPS fails, fall back to CPU with reduced batch size
        if device == "mps":
            print(f"\n[train] MPS training failed: {e}")
            print("[train] Falling back to CPU with batch=8...")
            device = "cpu"
            results = model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=min(args.batch, 8),
                device=device,
                name=args.name,
                patience=20,
                save=True,
                save_period=-1,
                close_mosaic=args.close_mosaic,
                cls=args.cls,
                **augmentation,
            )
        else:
            raise

    elapsed = time.time() - start_time
    print(f"\n[train] Training completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Print final metrics
    if results and hasattr(results, "results_dict"):
        print("\n[train] Final metrics:")
        for k, v in results.results_dict.items():
            print(f"  {k}: {v:.4f}")

    # Find and copy best.pt to models/best.pt
    # ultralytics saves to its own project root (may differ from CWD if run from a worktree).
    # Try the trainer's save_dir first, then fall back to relative search paths.
    best_pt = None

    # Method 1: use save_dir from trainer (available on the model after training)
    if hasattr(model, "trainer") and model.trainer and hasattr(model.trainer, "save_dir"):
        candidate = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
            print(f"[train] Found best.pt via trainer save_dir: {best_pt}")

    # Method 2: check relative path (works when CWD == project root)
    if best_pt is None:
        candidate = Path("runs/detect") / args.name / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate

    # Method 3: glob for suffixed run names (e.g., train2, train3)
    if best_pt is None:
        for base in [Path("runs/detect"), Path(__file__).resolve().parents[1] / "runs" / "detect"]:
            if base.exists():
                candidates = sorted(base.glob(f"{args.name}*/weights/best.pt"))
                if candidates:
                    best_pt = candidates[-1]
                    print(f"[train] Found best.pt at: {best_pt}")
                    break

    if best_pt is None:
        print(f"[train] WARNING: best.pt not found under runs/detect/{args.name}*/")
        return

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / "best.pt"

    # Strip optimizer state to reduce file size for inference/submission.
    # Training checkpoints include optimizer (~335MB for YOLOv8l) which is
    # only needed for training resume, not inference. The 420MB competition
    # submission limit requires stripping this.
    original_size = Path(best_pt).stat().st_size
    ckpt = torch.load(str(best_pt))
    ckpt["optimizer"] = None
    torch.save(ckpt, str(dest))
    stripped_size = dest.stat().st_size
    print(f"[train] Saved {best_pt} → {dest} (stripped optimizer)")
    print(f"[train] Original: {original_size / 1024 / 1024:.1f}MB → Stripped: {stripped_size / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    main()
