# NM i AI 2026 — NorgesGruppen Shelf Object Detection

## What This Is

A competition entry for the Norwegian AI Championship 2026 — NorgesGruppen shelf object detection task. Given JPEG images of grocery store shelves, the system detects every visible product with a bounding box and classifies it into one of 357 categories (356 products + unknown). The pipeline covers data preparation, model training, local evaluation, and sandbox-compliant submission packaging.

## Core Value

Maximize the competition score: `0.7 × detection_mAP + 0.3 × classification_mAP` at IoU≥0.5, within the 300-second inference timeout on an L4 GPU sandbox.

## Current State

Fresh start. No models trained, no submissions made. Training data is in `norgesgruppen_data/data/train/` (210 shelf images, 22.7k annotations, 357 categories). Product reference images in `norgesgruppen_data/data/test/` (327 products, multi-angle shots). No code exists yet.

## Architecture / Key Patterns

- **Training:** YOLOv8 (ultralytics) fine-tuned on COCO-format shelf annotations, M1 Max MPS/CPU backend
- **Inference:** ultralytics 8.1.0 on L4 GPU sandbox, Python 3.11, no network, blocked imports (os, subprocess, etc.)
- **Data format:** COCO annotations → YOLO format conversion for training
- **Submission:** `run.py` using pathlib for file ops, outputs COCO-format JSON predictions
- **Constraints:** 420MB weight limit, 3 weight files max, 10 Python files max, 300s timeout

## Capability Contract

See `.gsd/REQUIREMENTS.md` for the explicit capability contract, requirement status, and coverage mapping.

## Milestone Sequence

- [ ] M001: Competition Pipeline — End-to-end detection + classification pipeline with leaderboard submissions
