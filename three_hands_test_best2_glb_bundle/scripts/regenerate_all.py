#!/usr/bin/env python3
"""Regenerate every artifact using only files inside this bundle.

Run from any directory:
    python /path/to/three_hands_test_best2_glb_bundle/scripts/regenerate_all.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_overview(bundle_root: Path, samples: list[dict]) -> Path:
    tiles = []
    for sample in samples:
        frame_dir = bundle_root / sample["sample_dir"]
        image = cv2.imread(str(frame_dir / "previews" / "four_column.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame_dir / "previews" / "four_column.png")
        tiles.append(cv2.resize(image, (972, 271), interpolation=cv2.INTER_AREA))
    rows = []
    for index in range(0, len(tiles), 2):
        rows.append(np.hstack((tiles[index], tiles[index + 1])))
    overview = np.vstack(rows)
    output = bundle_root / "overview_six_examples.png"
    if not cv2.imwrite(str(output), overview):
        raise OSError(f"Could not write {output}")
    return output


def write_inventory(bundle_root: Path) -> Path:
    excluded = {"bundle_manifest.json", "validation_report.json"}
    records = []
    for path in sorted(item for item in bundle_root.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle_root).as_posix()
        if relative in excluded or "__pycache__" in path.parts:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "bundle": "three_hands_test_best2_glb_bundle",
        "excluded_generated_files": sorted(excluded),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    output = bundle_root / "bundle_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--opengl-platform", choices=("egl", "osmesa", "pyglet"), default="egl")
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parents[1]
    selection = json.loads((bundle_root / "selection_manifest.json").read_text(encoding="utf-8"))
    samples = selection["samples"]
    exporter = bundle_root / "scripts" / "export_gt_pred_glb.py"
    for sample in samples:
        frame_dir = bundle_root / sample["sample_dir"]
        command = [
            sys.executable, str(exporter),
            "--bundle-root", str(bundle_root),
            "--hand", sample["hand"],
            "--frame", str(sample["frame"]),
            "--camera-id", str(sample["camera_id"]),
            "--urdf", str(bundle_root / sample["urdf"]),
            "--metadata", str(frame_dir / "source" / "scene_gt.json"),
            "--prediction", str(frame_dir / "source" / "prediction.json"),
            "--rgb", str(frame_dir / "source" / "rgb.png"),
            "--depth", str(frame_dir / "source" / "depth.png"),
            "--output-dir", str(frame_dir),
            "--supersample", str(args.supersample),
            "--opengl-platform", args.opengl_platform,
        ]
        subprocess.run(command, check=True)
    overview = write_overview(bundle_root, samples)
    manifest = write_inventory(bundle_root)
    verifier = bundle_root / "scripts" / "verify_bundle.py"
    validation_report = bundle_root / "validation_report.json"
    subprocess.run([
        sys.executable, str(verifier), "--bundle-root", str(bundle_root),
        "--write-report", str(validation_report),
    ], check=True)
    print(json.dumps({
        "bundle_root": str(bundle_root), "sample_count": len(samples),
        "overview": str(overview), "manifest": str(manifest),
        "validation_report": str(validation_report),
    }, indent=2))


if __name__ == "__main__":
    main()
