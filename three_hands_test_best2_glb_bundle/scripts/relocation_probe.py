#!/usr/bin/env python3
"""Prove path independence by regenerating a temporary relocated bundle.

Run on a machine/node with EGL support:
    python /path/to/three_hands_test_best2_glb_bundle/scripts/relocation_probe.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.glob("*/frame_*/models/*.glb"))
    }


def load_regenerate_module(root: Path):
    path = root / "scripts" / "regenerate_all.py"
    spec = importlib.util.spec_from_file_location("portable_bundle_regenerate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--opengl-platform", choices=("egl", "osmesa", "pyglet"), default="egl")
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    original_hashes = output_hashes(source_root)
    if len(original_hashes) != 12:
        raise ValueError(f"Expected 12 source GLBs, found {len(original_hashes)}")

    with tempfile.TemporaryDirectory(prefix="dextrack_glb_relocation_") as temp_name:
        relocated_root = Path(temp_name) / "moved_bundle"
        shutil.copytree(
            source_root, relocated_root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        if relocated_root.resolve() == source_root.resolve():
            raise AssertionError("Relocation did not change the bundle path")
        subprocess.run([
            sys.executable, str(relocated_root / "scripts" / "regenerate_all.py"),
            "--supersample", str(args.supersample),
            "--opengl-platform", args.opengl_platform,
        ], check=True)
        subprocess.run([
            sys.executable, str(relocated_root / "scripts" / "verify_bundle.py"),
            "--bundle-root", str(relocated_root),
        ], check=True)
        relocated_hashes = output_hashes(relocated_root)
        mismatches = sorted(
            relative for relative, digest in original_hashes.items()
            if relocated_hashes.get(relative) != digest
        )
        extras = sorted(set(relocated_hashes) - set(original_hashes))
        if mismatches or extras:
            raise ValueError(f"Relocated GLB hashes changed; mismatches={mismatches}, extras={extras}")

    report = {
        "status": "PASS", "temporary_copy_was_at_a_different_path": True,
        "temporary_copy_removed_after_test": True,
        "regenerated_using_only_relocated_bundle_files": True,
        "relocated_bundle_verifier_status": "PASS",
        "compared_output_glb_count": len(original_hashes),
        "byte_identical_output_glb_count": len(original_hashes),
        "glb_hash_mismatch_count": 0,
        "glb_sha256": original_hashes,
    }
    report_path = source_root / "relocation_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    regenerate = load_regenerate_module(source_root)
    regenerate.write_inventory(source_root)
    subprocess.run([
        sys.executable, str(source_root / "scripts" / "verify_bundle.py"),
        "--bundle-root", str(source_root),
        "--write-report", str(source_root / "validation_report.json"),
    ], check=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
