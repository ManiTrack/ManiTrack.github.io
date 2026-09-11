#!/usr/bin/env python3
"""Run from any directory: python scripts/verify_bundle.py"""

from __future__ import annotations

import json
import os
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import trimesh


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (
    "model/inspire_hand_object.glb",
    "preview/inspire_hand_object_transparent.png",
    "metadata/result.json",
    "source/frame_676_scene.json",
    "source/camera_0_rgb_676.png",
    "source/camera_0_depth_676.png",
    "assets/inspire_hand/inspire_hand_left_visual.urdf",
    "scripts/generate_urdf_object_glb.py",
    "scripts/regenerate.py",
    "scripts/verify_bundle.py",
    "environment.txt",
    "README.md",
    "bundle_manifest.json",
)


def inside_bundle(path: Path) -> bool:
    return os.path.commonpath(
        [str(BUNDLE_ROOT.resolve()), str(path.resolve())]
    ) == str(BUNDLE_ROOT.resolve())


def scan_absolute_strings(value, where: str, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            scan_absolute_strings(item, f"{where}.{key}", found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan_absolute_strings(item, f"{where}[{index}]", found)
    elif isinstance(value, str):
        if value.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            found.append(f"{where}={value}")


def glb_json(path: Path) -> dict:
    payload = path.read_bytes()
    if payload[:4] != b"glTF":
        raise AssertionError("GLB magic header is missing")
    chunk_length, chunk_type = struct.unpack_from("<II", payload, 12)
    if chunk_type != 0x4E4F534A:
        raise AssertionError("First GLB chunk is not JSON")
    return json.loads(payload[20 : 20 + chunk_length].decode("utf-8"))


def main() -> None:
    missing = [name for name in EXPECTED if not (BUNDLE_ROOT / name).is_file()]
    if missing:
        raise AssertionError(f"Missing bundle files: {missing}")
    symlinks = [
        path.relative_to(BUNDLE_ROOT).as_posix()
        for path in BUNDLE_ROOT.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise AssertionError(f"Bundle contains symlinks: {symlinks}")

    urdf_path = BUNDLE_ROOT / "assets/inspire_hand/inspire_hand_left_visual.urdf"
    root = ET.parse(urdf_path).getroot()
    mesh_paths = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib["filename"]
        raw = Path(filename)
        if raw.is_absolute():
            raise AssertionError(f"Absolute URDF mesh path: {filename}")
        resolved = urdf_path.parent / raw
        if not inside_bundle(resolved) or not resolved.is_file():
            raise AssertionError(f"Invalid internal URDF mesh path: {filename}")
        mesh_paths.append(filename)
    if not mesh_paths:
        raise AssertionError("Visual-only URDF has no meshes")

    absolute_strings = []
    for json_path in BUNDLE_ROOT.rglob("*.json"):
        scan_absolute_strings(
            json.loads(json_path.read_text(encoding="utf-8")),
            json_path.relative_to(BUNDLE_ROOT).as_posix(),
            absolute_strings,
        )
    if absolute_strings:
        raise AssertionError(f"Absolute path strings remain: {absolute_strings}")

    glb_path = BUNDLE_ROOT / "model/inspire_hand_object.glb"
    document = glb_json(glb_path)
    external_uris = [
        item["uri"]
        for key in ("buffers", "images")
        for item in document.get(key, [])
        if "uri" in item and not str(item["uri"]).startswith("data:")
    ]
    if external_uris:
        raise AssertionError(f"GLB has external URIs: {external_uris}")
    scene = trimesh.load(glb_path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene) or len(scene.geometry) != 33:
        raise AssertionError("Expected 33 GLB geometry primitives")
    if not all(np.isfinite(geometry.vertices).all() for geometry in scene.geometry.values()):
        raise AssertionError("GLB contains non-finite vertices")
    materials = sorted(
        {
            str(getattr(getattr(g.visual, "material", None), "name", None))
            for g in scene.geometry.values()
        }
    )
    required_materials = {
        "black_plastic",
        "metal",
        "recorded_occluder_0_material",
        "white_plastic",
    }
    if not required_materials.issubset(set(materials)):
        raise AssertionError(f"Required PBR materials missing: {materials}")

    preview = cv2.imread(
        str(BUNDLE_ROOT / "preview/inspire_hand_object_transparent.png"),
        cv2.IMREAD_UNCHANGED,
    )
    if preview is None or preview.shape != (480, 480, 4):
        raise AssertionError(f"Unexpected transparent preview: {None if preview is None else preview.shape}")
    transparent = int(np.count_nonzero(preview[..., 3] == 0))
    visible = int(np.count_nonzero(preview[..., 3] > 0))
    if transparent == 0 or visible == 0:
        raise AssertionError("Preview does not contain both transparent and visible pixels")

    result = {
        "status": "SELF_CONTAINED_PASS",
        "bundle_root": str(BUNDLE_ROOT),
        "urdf_visual_references": len(mesh_paths),
        "unique_internal_visual_meshes": len(set(mesh_paths)),
        "glb_geometry_count": len(scene.geometry),
        "glb_external_uri_count": len(external_uris),
        "materials": materials,
        "transparent_pixels": transparent,
        "visible_pixels": visible,
        "absolute_json_path_count": len(absolute_strings),
        "symlink_count": len(symlinks),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
