#!/usr/bin/env python3
"""Convert COCO annotations to YOLO format with 80/20 train/val split.

Reads NorgesGruppen shelf annotations in COCO format and produces a YOLO-format
dataset directory with symlinked images, per-image label .txt files, and
dataset.yaml for ultralytics training.
"""

import json
import os
import random
import sys
from pathlib import Path

# --- Configuration ---
ANNOTATIONS_PATH = Path("/Users/jama/code/nmai/norgesgruppen_data/data/train/annotations.json")
IMAGES_DIR = Path("/Users/jama/code/nmai/norgesgruppen_data/data/train/images")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo_dataset"
SEED = 42
TRAIN_RATIO = 0.80


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert COCO [x, y, w, h] (top-left pixels) → YOLO [x_center, y_center, w, h] (normalized 0–1)."""
    x, y, w, h = bbox
    x_center = clamp((x + w / 2) / img_w)
    y_center = clamp((y + h / 2) / img_h)
    w_norm = clamp(w / img_w)
    h_norm = clamp(h / img_h)
    return x_center, y_center, w_norm, h_norm


def main():
    # --- Load COCO annotations ---
    if not ANNOTATIONS_PATH.exists():
        print(f"ERROR: Annotations file not found: {ANNOTATIONS_PATH}", file=sys.stderr)
        sys.exit(1)
    if not IMAGES_DIR.is_dir():
        print(f"ERROR: Images directory not found: {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    with open(ANNOTATIONS_PATH, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"Loaded: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")

    # --- Build lookup maps ---
    img_id_to_info = {img["id"]: img for img in images}

    # Group annotations by image_id
    img_id_to_anns: dict[int, list] = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # --- 80/20 train/val split at image level ---
    image_ids = [img["id"] for img in images]
    random.seed(SEED)
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * TRAIN_RATIO)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val (seed={SEED})")

    # --- Create output directories ---
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --- Write label files and create image symlinks ---
    total_anns_written = 0
    clamped_count = 0

    for img_id in image_ids:
        info = img_id_to_info[img_id]
        file_name = info["file_name"]
        img_w = info["width"]
        img_h = info["height"]
        stem = Path(file_name).stem  # e.g. "img_00001"

        split = "train" if img_id in train_ids else "val"

        # Symlink image
        src = IMAGES_DIR / file_name
        dst = OUTPUT_DIR / "images" / split / file_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(str(src.resolve()), str(dst))

        # Write label file
        anns = img_id_to_anns.get(img_id, [])
        label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
        with open(label_path, "w") as lf:
            for ann in anns:
                cat_id = ann["category_id"]
                bbox = ann["bbox"]

                # Check raw values before normalization
                raw_xc = (bbox[0] + bbox[2] / 2) / img_w
                raw_yc = (bbox[1] + bbox[3] / 2) / img_h
                raw_wn = bbox[2] / img_w
                raw_hn = bbox[3] / img_h
                if not (0 <= raw_xc <= 1 and 0 <= raw_yc <= 1 and 0 <= raw_wn <= 1 and 0 <= raw_hn <= 1):
                    clamped_count += 1

                xc, yc, wn, hn = coco_bbox_to_yolo(bbox, img_w, img_h)
                lf.write(f"{cat_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                total_anns_written += 1

    if clamped_count > 0:
        print(f"WARNING: {clamped_count} bbox values were clamped to [0,1]")

    # --- Generate dataset.yaml ---
    # Build names dict preserving category id → name mapping
    names = {cat["id"]: cat["name"] for cat in categories}

    yaml_content = f"path: {OUTPUT_DIR.resolve()}\n"
    yaml_content += "train: images/train\n"
    yaml_content += "val: images/val\n"
    yaml_content += f"nc: {len(categories)}\n"
    yaml_content += "names:\n"
    for cat_id in sorted(names.keys()):
        # Escape quotes in category names for YAML safety
        escaped = names[cat_id].replace('"', '\\"')
        yaml_content += f'  {cat_id}: "{escaped}"\n'

    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    # --- Summary ---
    train_labels = len(list((OUTPUT_DIR / "labels" / "train").glob("*.txt")))
    val_labels = len(list((OUTPUT_DIR / "labels" / "val").glob("*.txt")))
    train_images = len(list((OUTPUT_DIR / "images" / "train").iterdir()))
    val_images = len(list((OUTPUT_DIR / "images" / "val").iterdir()))

    print(f"\n=== Conversion Complete ===")
    print(f"Total images: {len(images)}")
    print(f"Train: {train_labels} labels, {train_images} images")
    print(f"Val:   {val_labels} labels, {val_images} images")
    print(f"Total annotations written: {total_anns_written}")
    print(f"Categories (nc): {len(categories)}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"dataset.yaml: {OUTPUT_DIR.resolve() / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
