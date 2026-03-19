# NM i AI 2026 — NorgesGruppen Shelf Object Detection

> **Competition:** Norwegian AI Championship 2026
> **Task:** Detect and classify grocery products on store shelves
> **Deadline:** March 22, 2026 at 15:00 CET (69-hour hackathon)
> **Submission:** Upload `.zip` containing `run.py` + model weights
> **Scoring:** `0.7 × detection_mAP + 0.3 × classification_mAP` (mAP@0.5)
> **Prize pool:** 1,000,000 NOK (shared across 3 tasks, each worth 33%)

---

## Problem Statement

Given JPEG images of Norwegian grocery store shelves, produce bounding box predictions for every visible product. Each prediction must include a confidence score and a product category ID (0–355). An `unknown_product` category exists at ID 356.

The score is a weighted combination:
- **70% detection mAP** — did you locate products? (IoU ≥ 0.5, category ignored)
- **30% classification mAP** — did you identify the correct product? (IoU ≥ 0.5 AND correct `category_id`)

A detection-only submission (all `category_id: 0`) can score a maximum of 0.70.

---

## Repository Structure

```
.
├── README.md
├── data/
│   ├── training/               # Extract NM_NGD_coco_dataset.zip here
│   │   ├── images/             # 248 shelf images (img_XXXXX.jpg)
│   │   └── annotations.json    # COCO-format annotations (~22.7k boxes, 357 categories)
│   └── product_images/         # Extract NM_NGD_product_images.zip here
│       ├── {barcode}/main.jpg  # Multi-angle product reference photos
│       ├── {barcode}/front.jpg
│       └── metadata.json       # Product names and annotation counts
├── src/
│   └── run.py                  # Entry point for submission (must be at zip root)
├── models/                     # Trained model weights
├── notebooks/                  # Exploration / EDA
├── scripts/                    # Training scripts, export scripts, local eval
└── submission/                 # Build area for creating submission .zip
```

---

## Training Data

### COCO Dataset (`data/training/`)

- **248 images** from 4 store sections: `Egg`, `Frokost`, `Knekkebrod`, `Varmedrikker`
- **~22,700 bounding box annotations** across **357 categories** (IDs 0–356)
- Annotation format: COCO (`annotations.json`)

```json
{
  "images": [
    {"id": 1, "file_name": "img_00001.jpg", "width": 2000, "height": 1500}
  ],
  "categories": [
    {"id": 0, "name": "VESTLANDSLEFSA TØRRE 10STK 360G", "supercategory": "product"},
    {"id": 356, "name": "unknown_product", "supercategory": "product"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 42,
      "bbox": [x, y, width, height],
      "area": 25688,
      "iscrowd": 0,
      "product_code": "8445291513365",
      "product_name": "NESCAFE VANILLA LATTE 136G NESTLE",
      "corrected": true
    }
  ]
}
```

**Key details:**
- `bbox` is `[x, y, width, height]` in pixels (COCO format)
- `product_code` is the barcode
- `corrected: true` indicates manually verified annotations
- `nc=357` when training (356 products + 1 unknown)

### Product Reference Images (`data/product_images/`)

- **327 products** with multi-angle photos: `main.jpg`, `front.jpg`, `back.jpg`, `left.jpg`, `right.jpg`, `top.jpg`, `bottom.jpg`
- Organized by barcode: `{product_code}/main.jpg`
- Includes `metadata.json` with product names and annotation counts

---

## Submission Contract

### Entry Point

```bash
python run.py --input /data/images --output /output/predictions.json
```

- **Input:** Directory of JPEG shelf images (`img_XXXXX.jpg`)
- **Output:** JSON array written to `--output` path

### Output Format

```json
[
  {
    "image_id": 42,
    "category_id": 0,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | `int` | Numeric ID extracted from filename (`img_00042.jpg` → `42`) |
| `category_id` | `int` | Product category ID (0–355), or 0 for detection-only |
| `bbox` | `[x, y, w, h]` | Bounding box in COCO format (pixels) |
| `score` | `float` | Confidence score (0–1) |

### Zip Structure

```
submission.zip
├── run.py          # Required: entry point (MUST be at zip root, not in a subfolder)
├── model.onnx      # Optional: model weights (.pt, .onnx, .safetensors, .npy)
└── utils.py        # Optional: helper code
```

**Build command:**
```bash
cd submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```

### Limits

| Limit | Value |
|-------|-------|
| Max zip size (uncompressed) | 420 MB |
| Max files | 1000 |
| Max Python files | 10 |
| Max weight files (`.pt`, `.pth`, `.onnx`, `.safetensors`, `.npy`) | 3 |
| Max weight size total | 420 MB |
| Allowed file types | `.py`, `.json`, `.yaml`, `.yml`, `.cfg`, `.pt`, `.pth`, `.onnx`, `.safetensors`, `.npy` |
| Submissions per day | 3 |
| Submissions in-flight | 2 |
| Execution timeout | 300 seconds |

---

## Sandbox Environment

| Resource | Spec |
|----------|------|
| Python | 3.11 |
| CPU | 4 vCPU |
| Memory | 8 GB |
| GPU | NVIDIA L4 (24 GB VRAM) |
| CUDA | 12.4 |
| Network | **None** (fully offline) |

### Pre-installed Packages

| Package | Version |
|---------|---------|
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| ultralytics | 8.1.0 |
| onnxruntime-gpu | 1.20.0 |
| opencv-python-headless | 4.9.0.80 |
| albumentations | 1.3.1 |
| Pillow | 10.2.0 |
| numpy | 1.26.4 |
| scipy | 1.12.0 |
| scikit-learn | 1.4.0 |
| pycocotools | 2.0.7 |
| ensemble-boxes | 1.0.9 |
| timm | 0.9.12 |
| supervision | 0.18.0 |
| safetensors | 0.4.2 |

**Cannot `pip install` at runtime.** If your framework isn't pre-installed, export to ONNX.

### Security Restrictions

**Blocked imports:** `os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`, `pickle`, `marshal`, `shelve`, `shutil`, `yaml`, `requests`, `urllib`, `http.client`, `multiprocessing`, `threading`, `signal`, `gc`, `code`, `codeop`, `pty`

**Blocked calls:** `eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous names

**Use `pathlib` instead of `os` for file operations. Use `json` instead of `yaml` for configs.**

---

## Approach Strategy

### Phase 1: Detection Baseline (target: ~0.50–0.60 score)
1. Fine-tune YOLOv8m/l on the COCO training data with `nc=357`
2. Pin `ultralytics==8.1.0` to match sandbox
3. Validate locally with a train/val split
4. Submit `.pt` weights directly (no ONNX needed for ultralytics)

### Phase 2: Improve Detection (target: ~0.60–0.70)
- Try larger models: YOLOv8x, RT-DETR-l/x
- Data augmentation (albumentations): mosaic, mixup, scale, color jitter
- Multi-scale inference / test-time augmentation (TTA)
- Ensemble multiple models using `ensemble-boxes` (WBF)

### Phase 3: Add Classification (target: 0.70+)
- Product reference images can augment training data or enable few-shot classification
- Two-stage pipeline: detect → crop → classify using product reference embeddings
- Fine-tune classification head on cropped product patches

### Key Considerations
- **248 images is small** — augmentation and preventing overfitting are critical
- **356 categories with only ~22.7k boxes** — long-tail distribution likely, some categories will have very few examples
- **Product reference images** are the secret weapon for classification — 327 products with multi-angle views
- **300s timeout** with L4 GPU — larger models are feasible but test inference time
- **FP16 recommended** — smaller weights and faster inference on L4

---

## Local Development

### Training Setup

```bash
# For YOLOv8 approach (recommended starting point)
pip install ultralytics==8.1.0

# For GPU training
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### Local Evaluation

```bash
# Run inference locally
python src/run.py --input data/training/images --output predictions.json

# Evaluate with pycocotools against ground truth
python scripts/evaluate.py --gt data/training/annotations.json --pred predictions.json
```

### Building Submission

```bash
# Copy necessary files to submission/
cp src/run.py submission/
cp models/best.pt submission/

# Create zip
cd submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"

# Verify structure (run.py must be at root)
unzip -l ../submission.zip | head -10
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `run.py not found at zip root` | Zip the **contents**, not the folder |
| `Disallowed file type: __MACOSX/...` | Use terminal: `zip -r ../sub.zip . -x ".*" "__MACOSX/*"` |
| `Disallowed file type: .bin` | Rename `.bin` → `.pt` (same format) or convert to `.safetensors` |
| `Security scan found violations` | Remove imports of `subprocess`, `socket`, `os`, etc. Use `pathlib` |
| `No predictions.json in output` | Ensure `run.py` writes to the `--output` path |
| `Timed out after 300s` | Ensure GPU is used (`model.to("cuda")`), or use a smaller model |
| `Exit code 137` | OOM (8 GB limit). Reduce batch size or use FP16 |
| `Exit code 139` | Segfault — likely version mismatch. Re-export with matching version or use ONNX |
| `ModuleNotFoundError` | Package not in sandbox. Export to ONNX or include model code in `.py` files |

---

## Reference Links

- **Submit:** https://app.ainm.no/submit/norgesgruppen-data
- **Docs:** https://app.ainm.no/docs/norgesgruppen-data/overview
- **Leaderboard:** https://app.ainm.no/leaderboard
- **Rules:** https://app.ainm.no/rules
- **MCP Server:** `claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp`
