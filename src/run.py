"""Sandbox-compliant YOLO inference for NorgesGruppen shelf detection.

Produces COCO-format predictions JSON from a directory of shelf images.
Designed for the competition sandbox: Python 3.11, PyTorch 2.6.0+cu124,
ultralytics 8.1.0, numpy 1.26.4, NVIDIA L4 GPU.

No blocked imports (os, sys, subprocess, socket, shutil, yaml, etc.).
All file operations use pathlib.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Compatibility patches — MUST run before any ultralytics import
# ---------------------------------------------------------------------------

# numpy.trapz was removed in numpy 2.0+; ultralytics 8.1.0 still uses it
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# PyTorch 2.6+ defaults to weights_only=True which breaks ultralytics 8.1.0
# model loading.  Patch to default weights_only=False.
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO  # noqa: E402 — must import after patches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CONFIDENCE_THRESHOLD = 0.001  # maximize recall
IMAGE_SIZE = 1280


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on shelf images and produce COCO-format predictions."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write predictions JSON.",
    )
    return parser.parse_args()


def extract_image_id(filename_stem: str) -> int:
    """Extract integer image_id from filename like img_00042 → 42."""
    return int(filename_stem.split("_")[1])


def xyxy_to_xywh(box):
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h] in pixels."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def main():
    args = parse_args()

    # Locate model weights relative to this script (zip extracts flat in sandbox)
    model_path = Path(__file__).parent / "best.pt"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}", flush=True)
        raise SystemExit(1)

    # Collect image paths
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}", flush=True)
        raise SystemExit(1)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"WARNING: No images found in {input_dir}", flush=True)
        # Write empty predictions array
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]")
        raise SystemExit(0)

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Images: {len(image_paths)} | Model: {model_path}", flush=True)

    # Load model
    model = YOLO(str(model_path))

    # Inference loop
    predictions = []
    t0 = time.time()

    for idx, img_path in enumerate(image_paths):
        try:
            image_id = extract_image_id(img_path.stem)
        except (IndexError, ValueError):
            print(f"WARNING: Cannot parse image_id from {img_path.name}, skipping", flush=True)
            continue

        try:
            # TTA (augment=True) was evaluated but not applied — it reduced weighted
            # score by 0.0019 on the epoch-3 checkpoint (0.2317 vs 0.2336 baseline).
            # On a fully-trained model, TTA may help; re-evaluate after training completes.
            results = model.predict(
                source=str(img_path),
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            print(f"ERROR: Inference failed on {img_path.name}: {exc}", flush=True)
            continue

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()

            for box, cls_id, score in zip(xyxy, classes, scores):
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                })

        # Progress every 10 images or on last image
        if (idx + 1) % 10 == 0 or idx == len(image_paths) - 1:
            elapsed = time.time() - t0
            print(
                f"[{idx + 1}/{len(image_paths)}] {img_path.name} — "
                f"{len(predictions)} predictions so far ({elapsed:.1f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(
        f"Done: {len(predictions)} predictions from {len(image_paths)} images "
        f"in {elapsed:.1f}s → {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
