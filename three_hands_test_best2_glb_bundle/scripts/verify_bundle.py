#!/usr/bin/env python3
"""Read-only structural and numerical verification for the portable bundle.

Run from any directory:
    python /path/to/three_hands_test_best2_glb_bundle/scripts/verify_bundle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import trimesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def absolute_strings(value, location: str = "$") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(absolute_strings(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(absolute_strings(item, f"{location}[{index}]"))
    elif isinstance(value, str):
        if value.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            found.append(f"{location}={value}")
    return found


def glb_info(path: Path) -> dict:
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise ValueError(f"Invalid GLB header: {path}")
    json_length, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing JSON chunk: {path}")
    document = json.loads(raw[20:20 + json_length].decode("utf-8").rstrip(" \t\r\n\x00"))
    uris = []
    for collection in ("buffers", "images"):
        uris.extend(item["uri"] for item in document.get(collection, []) if "uri" in item)
    loaded = trimesh.load(path, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError(f"Empty GLB scene: {path}")
    if not all(np.isfinite(geometry.vertices).all() for geometry in loaded.geometry.values()):
        raise ValueError(f"Non-finite GLB vertices: {path}")
    return {"bytes": len(raw), "sha256": sha256(path),
            "geometry_count": len(loaded.geometry), "external_uris": uris}


def verify_urdf(path: Path, root: Path) -> dict:
    tree = ET.parse(path)
    references = []
    for mesh in tree.findall(".//visual/geometry/mesh"):
        filename = mesh.attrib["filename"]
        mesh_path = (path.parent / filename).resolve()
        if Path(filename).is_absolute() or not inside(mesh_path, root):
            raise ValueError(f"URDF mesh escapes bundle: {filename}")
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        references.append(mesh_path.relative_to(root).as_posix())
    if not references:
        raise ValueError(f"No visual meshes in {path}")
    if tree.findall(".//collision") or tree.findall(".//inertial"):
        raise ValueError(f"Portable visual URDF still includes collision/inertial data: {path}")
    return {"path": path.relative_to(root).as_posix(),
            "visual_reference_count": len(references),
            "unique_visual_files": len(set(references))}


def verify_inventory(root: Path) -> dict:
    manifest_path = root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in manifest["files"]}
    excluded = set(manifest["excluded_generated_files"])
    current = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or "__pycache__" in path.parts:
            continue
        current[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if set(expected) != set(current):
        raise ValueError(
            f"Inventory file set mismatch; missing={sorted(set(expected) - set(current))}, "
            f"extra={sorted(set(current) - set(expected))}"
        )
    mismatches = []
    for relative, item in current.items():
        reference = expected[relative]
        if item["bytes"] != reference["bytes"] or item["sha256"] != reference["sha256"]:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"Inventory hash mismatch: {mismatches}")
    return {"file_count": len(current),
            "total_bytes": sum(item["bytes"] for item in current.values()),
            "hash_mismatch_count": 0}


def verify_sample(root: Path, sample: dict) -> dict:
    frame_dir = root / sample["sample_dir"]
    source_files = [frame_dir / "source" / name for name in
                    ("rgb.png", "depth.png", "scene_gt.json", "prediction.json")]
    if not all(path.is_file() for path in source_files):
        raise FileNotFoundError(f"Missing bundled source in {frame_dir}")
    source_rgb = cv2.imread(str(frame_dir / "source" / "rgb.png"), cv2.IMREAD_COLOR)
    if source_rgb is None:
        raise ValueError(f"Could not read source RGB: {frame_dir}")
    source_height, source_width = source_rgb.shape[:2]
    result_path = frame_dir / "metadata" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metric = result["pose_validation"]
    if metric["gt_fk_max_error_mm"] >= 0.1:
        raise ValueError(f"GT FK validation failed for {sample['sample_dir']}")
    if metric["report_vs_current_gt_max_drift_mm"] >= 0.0001:
        raise ValueError(f"Report/test drift for {sample['sample_dir']}")
    if metric["metric_delta_mm"] >= 0.0001:
        raise ValueError(f"Prediction metric mismatch for {sample['sample_dir']}")
    if abs(metric["reported_mean_gt_3d_error_mm"] - sample["mean_gt_3d_error_mm"]) >= 1e-5:
        raise ValueError(f"Selection metric mismatch for {sample['sample_dir']}")

    glbs = {}
    for kind in ("gt", "prediction"):
        path = frame_dir / "models" / f"{kind}.glb"
        info = glb_info(path)
        if info["external_uris"]:
            raise ValueError(f"External GLB URI in {path}: {info['external_uris']}")
        glbs[kind] = info

    alpha_stats = {}
    for kind in ("gt", "prediction"):
        path = frame_dir / "previews" / f"{kind}_transparent.png"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (source_height, source_width, 4):
            raise ValueError(f"Unexpected transparent preview shape: {path}, {None if image is None else image.shape}")
        transparent = int(np.count_nonzero(image[..., 3] == 0))
        visible = int(np.count_nonzero(image[..., 3] > 0))
        if not transparent or not visible:
            raise ValueError(f"Preview does not mix alpha/geometry: {path}")
        alpha_stats[kind] = {"transparent_pixels": transparent, "visible_pixels": visible}
    skeleton = cv2.imread(str(frame_dir / "previews" / "prediction_2d_skeleton.png"))
    four_column = cv2.imread(str(frame_dir / "previews" / "four_column.png"))
    if skeleton is None or skeleton.shape != (source_height, source_width, 3):
        raise ValueError(f"Unexpected skeleton preview: {sample['sample_dir']}")
    if four_column is None or four_column.shape != (542, 1944, 3):
        raise ValueError(f"Unexpected four-column preview: {sample['sample_dir']}")
    return {
        "hand": sample["hand"], "frame": sample["frame"],
        "rank_within_hand": sample["rank_within_hand"],
        "mean_gt_3d_error_mm": metric["recomputed_mean_gt_3d_error_mm"],
        "gt_fk_max_error_mm": metric["gt_fk_max_error_mm"],
        "report_gt_max_drift_mm": metric["report_vs_current_gt_max_drift_mm"],
        "recorded_object_count": len(result["recorded_objects_shared_by_gt_and_prediction"]),
        "glb": glbs, "alpha": alpha_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=None)
    parser.add_argument("--write-report", type=Path, default=None)
    args = parser.parse_args()
    root = (args.bundle_root or Path(__file__).resolve().parents[1]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    symlinks = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Bundle contains symlinks: {symlinks}")
    json_absolute_paths = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "validation_report.json":
            continue
        found = absolute_strings(json.loads(path.read_text(encoding="utf-8")))
        if found:
            json_absolute_paths[path.relative_to(root).as_posix()] = found
    if json_absolute_paths:
        raise ValueError(f"Absolute strings in JSON: {json_absolute_paths}")
    forbidden_text = []
    forbidden_home_prefix = "/public" + "/home/"
    for path in list((root / "scripts").glob("*.py")) + [root / "README.md"]:
        text = path.read_text(encoding="utf-8")
        if forbidden_home_prefix in text or re.search(r"[A-Za-z]:\\", text):
            forbidden_text.append(path.relative_to(root).as_posix())
    if forbidden_text:
        raise ValueError(f"Hard-coded external filesystem paths: {forbidden_text}")

    selection = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
    if len(selection["samples"]) != 6:
        raise ValueError("Expected exactly six samples")
    hand_counts = {hand: sum(item["hand"] == hand for item in selection["samples"])
                   for hand in ("inspire", "xhand", "leap")}
    if hand_counts != {"inspire": 2, "xhand": 2, "leap": 2}:
        raise ValueError(f"Wrong per-hand sample counts: {hand_counts}")
    urdfs = [verify_urdf(root / f"assets/{hand}/hand_visual.urdf", root)
             for hand in ("inspire", "xhand", "leap")]
    asset_glbs = [glb_info(path) for path in sorted((root / "assets").rglob("*.glb"))]
    if any(item["external_uris"] for item in asset_glbs):
        raise ValueError("A bundled source visual GLB contains an external URI")
    samples = [verify_sample(root, sample) for sample in selection["samples"]]
    overview = cv2.imread(str(root / "overview_six_examples.png"))
    if overview is None or overview.shape != (813, 1944, 3):
        raise ValueError(f"Unexpected overview shape: {None if overview is None else overview.shape}")
    inventory = verify_inventory(root)
    report = {
        "status": "PASS", "self_contained": True,
        "sample_count": len(samples), "hand_counts": hand_counts,
        "symlink_count": 0, "absolute_json_path_count": 0,
        "hard_coded_external_path_count": 0,
        "output_glb_count": 12,
        "all_output_glbs_have_zero_external_uris": True,
        "bundled_asset_glb_count": len(asset_glbs),
        "all_bundled_asset_glbs_have_zero_external_uris": True,
        "visual_urdfs": urdfs, "samples": samples,
        "overview_shape_hwc": list(overview.shape), "inventory": inventory,
    }
    if args.write_report is not None:
        output = args.write_report.resolve()
        if not inside(output, root):
            raise ValueError("Validation report path must stay inside bundle")
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
