#!/usr/bin/env python3
"""Build submission.zip for competition upload.

Copies src/run.py and models/best.pt into a flat zip at the repo root,
then validates all competition constraints before finishing.

Competition constraints:
  - Max zip uncompressed: 420 MB (440,401,920 bytes)
  - Max Python files: 10
  - Max weight files (.pt, .pth, .onnx, .safetensors, .npy): 3
  - Max total files: 1000
  - Allowed types: .py .json .yaml .yml .cfg .pt .pth .onnx .safetensors .npy
  - run.py MUST be at zip root (not in a subfolder)
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_RUN_PY = REPO_ROOT / "src" / "run.py"
SRC_WEIGHTS = REPO_ROOT / "models" / "best.pt"
OUTPUT_ZIP = REPO_ROOT / "submission.zip"

ALLOWED_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".cfg",
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}

MAX_UNCOMPRESSED_BYTES = 420 * 1024 * 1024  # 440,401,920 bytes (420 MB)
MAX_WEIGHT_FILES = 3
MAX_PYTHON_FILES = 10
MAX_TOTAL_FILES = 1000


def build_zip() -> Path:
    """Build submission.zip from source files and return its path."""
    # Validate source files exist
    for src, label in [(SRC_RUN_PY, "src/run.py"), (SRC_WEIGHTS, "models/best.pt")]:
        if not src.exists():
            print(f"ERROR: Source file not found: {label} ({src})", flush=True)
            sys.exit(1)

    # Stage files in a temp directory to avoid any path issues
    staging_dir = tempfile.mkdtemp(prefix="submission_staging_")
    try:
        staging = Path(staging_dir)
        shutil.copy2(str(SRC_RUN_PY), str(staging / "run.py"))
        shutil.copy2(str(SRC_WEIGHTS), str(staging / "best.pt"))

        # Build the zip
        with zipfile.ZipFile(str(OUTPUT_ZIP), "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(staging.iterdir()):
                # arcname = just the filename → flat structure at zip root
                zf.write(str(f), arcname=f.name)

        print(f"Built: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 1024 / 1024:.1f} MB compressed)", flush=True)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return OUTPUT_ZIP


def validate_zip(zip_path: Path) -> bool:
    """Validate submission.zip against competition constraints. Returns True if all pass."""
    print("\n=== Submission Validation ===", flush=True)
    all_pass = True

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        entries = zf.infolist()
        names = [e.filename for e in entries]
        total_uncompressed = sum(e.file_size for e in entries)

        # --- Check 1: run.py exists at root ---
        has_run_py = "run.py" in names
        status = "PASS" if has_run_py else "FAIL"
        print(f"  [{status}] run.py at root: {'found' if has_run_py else 'MISSING'}", flush=True)
        all_pass &= has_run_py

        # --- Check 2: No subdirectories (all files at root) ---
        has_subdirs = any("/" in name for name in names)
        status = "PASS" if not has_subdirs else "FAIL"
        print(f"  [{status}] Flat structure (no subdirs): {'yes' if not has_subdirs else 'NO — ' + str([n for n in names if '/' in n])}", flush=True)
        all_pass &= not has_subdirs

        # --- Check 3: Total uncompressed size ---
        size_ok = total_uncompressed <= MAX_UNCOMPRESSED_BYTES
        status = "PASS" if size_ok else "FAIL"
        print(f"  [{status}] Uncompressed size: {total_uncompressed / 1024 / 1024:.1f} MB (limit: 420 MB)", flush=True)
        all_pass &= size_ok

        # --- Check 4: Weight file count ---
        weight_files = [n for n in names if Path(n).suffix.lower() in WEIGHT_EXTENSIONS]
        wt_ok = len(weight_files) <= MAX_WEIGHT_FILES
        status = "PASS" if wt_ok else "FAIL"
        print(f"  [{status}] Weight files: {len(weight_files)} (limit: {MAX_WEIGHT_FILES}) — {weight_files}", flush=True)
        all_pass &= wt_ok

        # --- Check 5: Python file count ---
        py_files = [n for n in names if n.endswith(".py")]
        py_ok = len(py_files) <= MAX_PYTHON_FILES
        status = "PASS" if py_ok else "FAIL"
        print(f"  [{status}] Python files: {len(py_files)} (limit: {MAX_PYTHON_FILES}) — {py_files}", flush=True)
        all_pass &= py_ok

        # --- Check 6: Total file count ---
        total_ok = len(names) <= MAX_TOTAL_FILES
        status = "PASS" if total_ok else "FAIL"
        print(f"  [{status}] Total files: {len(names)} (limit: {MAX_TOTAL_FILES})", flush=True)
        all_pass &= total_ok

        # --- Check 7: All extensions allowed ---
        disallowed = [n for n in names if Path(n).suffix.lower() not in ALLOWED_EXTENSIONS]
        ext_ok = len(disallowed) == 0
        status = "PASS" if ext_ok else "FAIL"
        print(f"  [{status}] Allowed extensions only: {'yes' if ext_ok else 'NO — ' + str(disallowed)}", flush=True)
        all_pass &= ext_ok

    # --- Summary ---
    print(f"\n--- Summary ---", flush=True)
    print(f"  Files: {len(names)}", flush=True)
    print(f"  Uncompressed size: {total_uncompressed / 1024 / 1024:.1f} MB", flush=True)
    print(f"  Weight files: {len(weight_files)}", flush=True)
    print(f"  Python files: {len(py_files)}", flush=True)
    print(f"  Overall: {'ALL CHECKS PASSED' if all_pass else 'VALIDATION FAILED'}", flush=True)

    return all_pass


def main():
    zip_path = build_zip()
    if validate_zip(zip_path):
        print(f"\n✓ submission.zip is ready for upload", flush=True)
        sys.exit(0)
    else:
        print(f"\n✗ submission.zip FAILED validation — fix issues above", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
