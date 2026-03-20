#!/usr/bin/env python3
"""Evaluate a YOLO model on the NorgesGruppen val set using pycocotools.

Computes:
  - Detection mAP@0.5 (category-agnostic: all boxes mapped to a single class)
  - Classification mAP@0.5 (category-aware: box must match location AND class)
  - Weighted score: 0.7 × detection_mAP + 0.3 × classification_mAP

Uses the original COCO annotations.json as ground truth, filtered to the val
split produced by convert_coco_to_yolo.py (seed=42, 80/20 split).
"""

import argparse
import copy
import functools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# --- Compatibility patches for ultralytics 8.1.0 + numpy 2.x + PyTorch 2.6+ ---
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# --- Constants ---
ANNOTATIONS_PATH = Path(__file__).resolve().parent.parent / "norgesgruppen_data" / "data" / "train" / "annotations.json"
SEED = 42
TRAIN_RATIO = 0.80


def get_val_image_ids(annotations_path: Path) -> set[int]:
    """Reproduce the exact val split from convert_coco_to_yolo.py."""
    with open(annotations_path) as f:
        coco_data = json.load(f)
    image_ids = [img["id"] for img in coco_data["images"]]
    random.seed(SEED)
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * TRAIN_RATIO)
    return set(image_ids[split_idx:])


def build_gt_coco(annotations_path: Path, val_ids: set[int]) -> COCO:
    """Build a COCO object from annotations.json filtered to val images only."""
    with open(annotations_path) as f:
        coco_data = json.load(f)

    gt_data = {
        "images": [img for img in coco_data["images"] if img["id"] in val_ids],
        "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in val_ids],
        "categories": coco_data["categories"],
    }

    # pycocotools COCO() can load from a file or from a dataset dict
    # We use the internal method to avoid writing a temp file
    coco_gt = COCO()
    coco_gt.dataset = gt_data
    coco_gt.createIndex()
    return coco_gt


def run_inference(model: YOLO, image_dir: Path, conf: float, imgsz: int, verbose: bool, augment: bool = False) -> list[dict]:
    """Run model inference on all images in image_dir, return COCO-format predictions."""
    predictions = []
    image_paths = sorted(image_dir.iterdir())

    for i, img_path in enumerate(image_paths):
        if not img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            continue

        # Extract image_id from filename: img_00042.jpg → 42
        image_id = int(img_path.stem.split("_")[1])

        results = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, verbose=False, augment=augment)
        r = results[0]

        if len(r.boxes) == 0:
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy pixel format
        classes = r.boxes.cls.cpu().numpy().astype(int)  # [N]
        scores = r.boxes.conf.cpu().numpy()  # [N]

        for box, cls_id, score in zip(boxes_xyxy, classes, scores):
            x1, y1, x2, y2 = box
            # Convert xyxy → COCO [x, y, w, h]
            w = x2 - x1
            h = y2 - y1
            predictions.append({
                "image_id": image_id,
                "category_id": int(cls_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })

        if verbose and (i + 1) % 10 == 0:
            print(f"  Inference: {i + 1}/{len(image_paths)} images processed")

    return predictions


def compute_map(coco_gt: COCO, predictions: list[dict], category_agnostic: bool = False) -> float:
    """Compute mAP@0.5 using pycocotools COCOeval.

    If category_agnostic=True, remap all GT and prediction category_ids to 0
    so IoU matching ignores class labels (detection-only metric).
    """
    if not predictions:
        print("  WARNING: No predictions to evaluate")
        return 0.0

    if category_agnostic:
        # Deep copy GT and remap all categories to a single class
        gt_data = copy.deepcopy(coco_gt.dataset)
        gt_data["categories"] = [{"id": 0, "name": "object", "supercategory": "object"}]
        for ann in gt_data["annotations"]:
            ann["category_id"] = 0

        coco_gt_eval = COCO()
        coco_gt_eval.dataset = gt_data
        coco_gt_eval.createIndex()

        # Remap predictions to single class
        preds_eval = copy.deepcopy(predictions)
        for p in preds_eval:
            p["category_id"] = 0
    else:
        coco_gt_eval = coco_gt
        preds_eval = predictions

    # loadRes creates a result COCO object from the predictions list
    coco_dt = coco_gt_eval.loadRes(preds_eval)

    # Run evaluation
    coco_eval = COCOeval(coco_gt_eval, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.5])  # Evaluate at IoU=0.5 only

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = mAP across iouThrs (which is just 0.5 since we set a single threshold)
    # stats[1] = AP@IoU=0.5 explicitly — both should be identical
    map_50 = float(coco_eval.stats[1])
    return map_50


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on NorgesGruppen val set")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to evaluate (default: val)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)")
    parser.add_argument("--verbose", action="store_true", help="Print per-class AP and inference progress")
    parser.add_argument("--augment", action="store_true", help="Enable test-time augmentation (TTA) during inference")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset config
    import yaml
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    dataset_path = Path(data_cfg["path"])
    image_dir = dataset_path / data_cfg[args.split]
    if not image_dir.is_dir():
        print(f"ERROR: Image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    num_images = len(list(image_dir.iterdir()))
    print(f"[evaluate] Model: {args.model}")
    print(f"[evaluate] Dataset: {args.data}")
    print(f"[evaluate] Split: {args.split} ({num_images} images)")
    print(f"[evaluate] ImgSz: {args.imgsz}, Conf: {args.conf}")
    print(f"[evaluate] TTA (augment): {args.augment}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    model = YOLO(args.model)

    # Determine annotations path
    annotations_path = ANNOTATIONS_PATH
    if not annotations_path.exists():
        print(f"ERROR: Annotations not found: {annotations_path}", file=sys.stderr)
        sys.exit(1)

    # Get val image IDs (reproduce the same split as converter)
    val_ids = get_val_image_ids(annotations_path)
    print(f"[evaluate] Val image IDs: {len(val_ids)}")

    # Build ground truth COCO object filtered to val images
    coco_gt = build_gt_coco(annotations_path, val_ids)
    gt_ann_count = len(coco_gt.dataset["annotations"])
    print(f"[evaluate] Ground truth annotations (val): {gt_ann_count}")

    # Run inference
    print(f"[evaluate] Running inference on {num_images} images...")
    predictions = run_inference(model, image_dir, args.conf, args.imgsz, args.verbose, augment=args.augment)
    print(f"[evaluate] Total predictions: {len(predictions)}")

    if len(predictions) == 0:
        print("ERROR: No predictions generated. Check model and confidence threshold.", file=sys.stderr)
        sys.exit(1)

    # Compute detection mAP (category-agnostic)
    print("\n--- Detection mAP (category-agnostic) ---")
    det_map = compute_map(coco_gt, predictions, category_agnostic=True)

    # Compute classification mAP (category-aware)
    print("\n--- Classification mAP (category-aware) ---")
    cls_map = compute_map(coco_gt, predictions, category_agnostic=False)

    # Compute weighted score
    weighted = 0.7 * det_map + 0.3 * cls_map

    # Print summary
    print(f"\n{'=' * 40}")
    print(f"=== Evaluation Results ===")
    print(f"{'=' * 40}")
    print(f"Val images:              {num_images}")
    print(f"Predictions:             {len(predictions)}")
    print(f"GT annotations (val):    {gt_ann_count}")
    print(f"Detection mAP@0.5:       {det_map:.4f}")
    print(f"Classification mAP@0.5:  {cls_map:.4f}")
    print(f"Weighted Score:          0.7×{det_map:.4f} + 0.3×{cls_map:.4f} = {weighted:.4f}")
    print(f"{'=' * 40}")

    # Return exit code based on detection mAP threshold
    if det_map < 0.40:
        print(f"\nWARNING: detection_mAP ({det_map:.4f}) is below 0.40 target", file=sys.stderr)


if __name__ == "__main__":
    main()
