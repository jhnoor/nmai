# M001: Competition Pipeline

**Vision:** End-to-end object detection and classification pipeline for the NM i AI 2026 NorgesGruppen shelf detection challenge, maximizing the weighted score `0.7 × detection_mAP + 0.3 × classification_mAP`.

## Success Criteria

- At least one successful leaderboard submission with a non-trivial score
- Local evaluation produces mAP numbers that approximately match leaderboard feedback
- Detection mAP exceeds 0.40 (achievable baseline with YOLOv8m on 210 images)
- Classification enabled with nc=357, adding measurable score above detection-only
- Full pipeline is reproducible: data prep → train → eval → build zip → submit

## Key Risks / Unknowns

- **MPS training speed** — could be slower than CPU for 210-image dataset, blocking iteration speed
- **ultralytics 8.1.0 compatibility** — weights trained on a different version may fail in sandbox
- **Long-tail categories** — 74 categories with <5 examples will have near-zero per-class mAP
- **Overfitting on 210 images** — augmentation strategy is make-or-break

## Proof Strategy

- MPS vs CPU speed → retire in S01 by benchmarking both backends on first training run
- ultralytics compatibility → retire in S02 by submitting and getting a score (not a runtime error)
- Overfitting → retire in S01 by validating on held-out val set and checking train/val mAP gap

## Verification Classes

- Contract verification: local mAP evaluation, submission zip structure validation
- Integration verification: successful leaderboard submission with a score
- Operational verification: none
- UAT / human verification: leaderboard score review

## Milestone Definition of Done

This milestone is complete only when all are true:

- At least one leaderboard submission has received a score
- Local evaluation pipeline produces reliable mAP estimates
- Classification is enabled and contributing to the score
- Pipeline scripts are documented and reproducible
- Best achievable score has been submitted given time constraints

## Requirement Coverage

- Covers: R001, R002, R003, R004, R005, R006, R007, R008, R009, R010
- Partially covers: none
- Leaves for later: R011, R012 (deferred)
- Orphan risks: none

## Slices

- [x] **S01: Data Pipeline & Detection Baseline** `risk:high` `depends:[]`
  > After this: YOLOv8 trained on shelf data, local mAP evaluation produces detection score on val set.

- [x] **S02: Sandbox-Compliant Inference & First Submission** `risk:high` `depends:[S01]`
  > After this: run.py passes local smoke test, submission zip built, first leaderboard score received.

- [x] **S03: Detection Tuning & Classification** `risk:medium` `depends:[S02]`
  > After this: Improved model with classification enabled (nc=357), better leaderboard score than S02 baseline.

- [x] **S04: Score Maximization & Final Submissions** `risk:low` `depends:[S03]`
  > After this: TTA/augmentation tuning applied, best achievable score submitted before deadline.

## Boundary Map

### S01 → S02

Produces:
- `scripts/convert_coco_to_yolo.py` — data conversion script producing YOLO-format labels and dataset.yaml
- `scripts/train.py` — training script with configurable model size, epochs, augmentation
- `scripts/evaluate.py` — local evaluation computing weighted mAP score
- `models/best.pt` — trained YOLOv8 detection model weights
- `data/yolo_dataset/` — converted dataset with train/val split and dataset.yaml
- Benchmark result: MPS vs CPU training speed for this dataset

Consumes:
- nothing (first slice)

### S02 → S03

Produces:
- `src/run.py` — sandbox-compliant inference script (pathlib, no blocked imports)
- `scripts/build_submission.py` — zip builder with structure validation
- First leaderboard score — baseline to beat
- Confirmed ultralytics 8.1.0 compatibility (weights load in sandbox)

Consumes from S01:
- `models/best.pt` — trained model weights
- `scripts/evaluate.py` — local eval for pre-submission validation

### S03 → S04

Produces:
- Improved `models/best.pt` — tuned model with better augmentation, possibly larger architecture
- Classification-enabled model (nc=357) with measured classification mAP contribution
- Updated training config with best hyperparameters found

Consumes from S02:
- `src/run.py` — inference script (may need minor updates for new model)
- `scripts/build_submission.py` — submission builder
- Baseline leaderboard score to compare against

### S04 (terminal)

Produces:
- Final submission with best achievable score
- TTA-enabled run.py if it improves score within timeout budget

Consumes from S03:
- Best model weights
- Tuned training pipeline
- `src/run.py` and `scripts/build_submission.py`
