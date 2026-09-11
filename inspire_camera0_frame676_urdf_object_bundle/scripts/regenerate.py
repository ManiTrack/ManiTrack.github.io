#!/usr/bin/env python3
"""Run from any directory: python scripts/regenerate.py"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(BUNDLE_ROOT.resolve()).as_posix()


def write_inventory() -> None:
    files = {}
    for path in sorted(BUNDLE_ROOT.rglob("*")):
        if not path.is_file() or path.name == "bundle_manifest.json":
            continue
        files[relative(path)] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    payload = {
        "schema": "dextrack-self-contained-visualization-bundle-v1",
        "path_policy": "all listed paths are relative to this bundle root",
        "files": files,
    }
    (BUNDLE_ROOT / "bundle_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def sanitize_result_manifest(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    result["artifact"] = "model/inspire_hand_object.glb"
    result["inputs"]["urdf"] = "assets/inspire_hand/inspire_hand_left_visual.urdf"
    result["inputs"]["metadata"] = "source/frame_676_scene.json"
    result["inputs"]["rgb"] = "source/camera_0_rgb_676.png"
    result["inputs"]["depth"] = "source/camera_0_depth_676.png"
    result["transparent_preview"] = "preview/inspire_hand_object_transparent.png"
    result["alignment_check"] = "validation/camera0_alignment_check.png"
    result["bundle_relative_paths"] = True
    for item in result.get("objects", []):
        logical_path = item.pop("path", None)
        if logical_path:
            item["scene_prim"] = str(logical_path).lstrip("/")
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import generate_urdf_object_glb

    output = BUNDLE_ROOT / "model" / "inspire_hand_object.glb"
    preview = BUNDLE_ROOT / "preview" / "inspire_hand_object_transparent.png"
    result = BUNDLE_ROOT / "metadata" / "result.json"
    alignment = BUNDLE_ROOT / "validation" / "camera0_alignment_check.png"
    sys.argv = [
        "generate_urdf_object_glb.py",
        "--urdf",
        str(BUNDLE_ROOT / "assets" / "inspire_hand" / "inspire_hand_left_visual.urdf"),
        "--metadata",
        str(BUNDLE_ROOT / "source" / "frame_676_scene.json"),
        "--rgb",
        str(BUNDLE_ROOT / "source" / "camera_0_rgb_676.png"),
        "--depth",
        str(BUNDLE_ROOT / "source" / "camera_0_depth_676.png"),
        "--camera-id",
        "0",
        "--frame-file-index",
        "676",
        "--supersample",
        "2",
        "--opengl-platform",
        "egl",
        "--output",
        str(output),
        "--transparent-preview",
        str(preview),
        "--alignment-check",
        str(alignment),
        "--manifest",
        str(result),
    ]
    generate_urdf_object_glb.main()
    sanitize_result_manifest(result)
    write_inventory()
    print(f"SELF-CONTAINED REGENERATION PASS: {BUNDLE_ROOT}")


if __name__ == "__main__":
    main()
