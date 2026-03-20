# M001: Competition Pipeline — Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

## Project Description

NM i AI 2026 — NorgesGruppen shelf object detection competition entry. Detect and classify grocery products on store shelf images. 69-hour hackathon, deadline March 22 at 15:00 CET. Prize pool: 1,000,000 NOK shared across 3 tasks.

## Why This Milestone

This is the only milestone — it covers the entire competition pipeline from data preparation through final submission. Time pressure (≈40 hours remaining) demands a single focused milestone.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Run a training script that fine-tunes YOLOv8 on the shelf dataset
- Evaluate model performance locally with the competition's scoring formula
- Build and submit a zip to the competition leaderboard
- Iterate on model improvements and resubmit

### Entry point / environment

- Entry point: Python scripts (`train.py`, `evaluate.py`, `build_submission.py`, `run.py`)
- Environment: local dev on M1 Max 64GB (training), L4 GPU sandbox (inference)
- Live dependencies involved: competition submission endpoint (https://app.ainm.no/submit/norgesgruppen-data)

## Completion Class

- Contract complete means: local mAP evaluation shows reasonable scores, submission zip passes structure validation
- Integration complete means: at least one successful leaderboard submission with a score
- Operational complete means: none — this is a competition, not a deployed service

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- A submission has been accepted by the leaderboard and received a score
- Local evaluation matches approximately the leaderboard score (validating our eval is correct)
- The pipeline is reproducible: data prep → train → eval → submit

## Risks and Unknowns

- **MPS training speed** — MPS can be slower than CPU for small datasets on Apple Silicon. Must benchmark both backends early.
- **ultralytics version compatibility** — training locally with a different ultralytics version than sandbox (8.1.0) could cause weight loading failures. Pin version.
- **Long-tail category distribution** — 74 categories have <5 annotations. Classification mAP for rare classes will be near zero.
- **210 images is tiny** — high overfitting risk. Augmentation strategy is critical.
- **300s inference timeout** — larger models + TTA must stay within budget on L4 GPU.
- **3 submissions/day** — limited feedback loop. Local eval must be reliable.

## Existing Codebase / Prior Art

- `norgesgruppen_data/data/train/annotations.json` — COCO format, 22,731 annotations, 356 categories (ids 0-355, where 355=unknown_product)
- `norgesgruppen_data/data/train/images/` — 210 shelf images (various resolutions, mostly 4032x3024)
- `norgesgruppen_data/data/test/` — 344 product reference image directories (327 with images), organized by barcode
- `norgesgruppen_data/data/test/metadata.json` — product names, annotation counts, barcode→name mapping
- Category names in annotations match product names in metadata (323/324 exact matches) — this is the bridge for barcode→category_id mapping

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions — it is an append-only register; read it during planning, append to it during execution.

## Relevant Requirements

- R001–R010 (all active requirements map to this milestone)

## Scope

### In Scope

- COCO→YOLO data format conversion with train/val split
- YOLOv8 fine-tuning with nc=357 (detection + classification in one model)
- Local evaluation script with competition scoring formula
- Sandbox-compliant run.py using pathlib
- Submission zip builder with validation
- Training augmentation for small dataset
- Model size selection (n/s/m/l/x) balancing accuracy vs inference time
- TTA / multi-scale inference if time permits

### Out of Scope / Non-Goals

- Two-stage detect→classify pipeline (deferred — R011)
- Model ensembling (deferred — R012)
- Custom architectures beyond ultralytics built-ins (R013)
- Product reference image integration for classification
- Web scraping or additional training data

## Technical Constraints

- Sandbox: ultralytics 8.1.0, PyTorch 2.6.0+cu124, Python 3.11, NVIDIA L4 24GB, 8GB RAM, no network
- Blocked imports: os, sys, subprocess, socket, ctypes, builtins, importlib, pickle, marshal, shelve, shutil, yaml, requests, urllib, http.client, multiprocessing, threading, signal, gc, code, codeop, pty
- Max submission: 420MB uncompressed, 3 weight files, 10 Python files
- Training hardware: M1 Max 64GB, MPS or CPU backend
- Pin ultralytics==8.1.0 for training to match sandbox

## Integration Points

- **Competition leaderboard** — submission endpoint accepts zip, returns mAP scores
- **pycocotools** — local mAP evaluation against ground truth annotations
- **ultralytics** — training and inference framework, must match sandbox version 8.1.0

## Open Questions

- Whether MPS or CPU is faster for YOLOv8 training on this 210-image dataset — benchmark in S01
- Optimal model size (m vs l vs x) balancing accuracy and 300s inference timeout — test in S01/S03
- Whether classification adds enough score to justify nc=357 vs detection-only (nc=1) — compare in S03
