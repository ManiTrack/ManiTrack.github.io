#!/usr/bin/env python3
"""Create hand-only prediction GLBs without altering the bundled originals.

Run from any directory:
    python /path/to/three_hands_test_best2_glb_bundle/scripts/export_prediction_hand_only.py
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


GLB_HEADER = struct.Struct("<4sII")
GLB_CHUNK_HEADER = struct.Struct("<II")
JSON_CHUNK = 0x4E4F534A


def read_glb(path: Path):
    raw = path.read_bytes()
    magic, version, total_length = GLB_HEADER.unpack_from(raw, 0)
    if magic != b"glTF" or version != 2 or total_length != len(raw):
        raise ValueError(f"Invalid GLB header: {path}")

    chunks = []
    offset = GLB_HEADER.size
    while offset < len(raw):
        length, chunk_type = GLB_CHUNK_HEADER.unpack_from(raw, offset)
        start = offset + GLB_CHUNK_HEADER.size
        end = start + length
        if end > len(raw):
            raise ValueError(f"Invalid GLB chunk length: {path}")
        chunks.append((chunk_type, raw[start:end]))
        offset = end

    json_indices = [index for index, item in enumerate(chunks) if item[0] == JSON_CHUNK]
    if len(json_indices) != 1:
        raise ValueError(f"Expected one JSON chunk in {path}")
    json_index = json_indices[0]
    document = json.loads(
        chunks[json_index][1].decode("utf-8").rstrip(" \t\r\n\x00")
    )
    return chunks, json_index, document


def strip_recorded_objects(document: dict):
    nodes = document.get("nodes", [])
    removed_nodes = {
        index for index, node in enumerate(nodes)
        if str(node.get("name", "")).startswith("OBJECT__")
    }
    kept_node_indices = [index for index in range(len(nodes)) if index not in removed_nodes]
    node_map = {old: new for new, old in enumerate(kept_node_indices)}

    new_nodes = []
    for old_index in kept_node_indices:
        node = dict(nodes[old_index])
        if "children" in node:
            children = [node_map[index] for index in node["children"] if index in node_map]
            if children:
                node["children"] = children
            else:
                node.pop("children")
        new_nodes.append(node)

    for scene in document.get("scenes", []):
        scene["nodes"] = [node_map[index] for index in scene.get("nodes", []) if index in node_map]

    used_meshes = sorted({node["mesh"] for node in new_nodes if "mesh" in node})
    mesh_map = {old: new for new, old in enumerate(used_meshes)}
    meshes = document.get("meshes", [])
    removed_mesh_count = len(meshes) - len(used_meshes)
    document["meshes"] = [meshes[index] for index in used_meshes]
    for node in new_nodes:
        if "mesh" in node:
            node["mesh"] = mesh_map[node["mesh"]]
    document["nodes"] = new_nodes

    return len(removed_nodes), removed_mesh_count


def write_glb(path: Path, chunks, json_index: int, document: dict) -> None:
    json_bytes = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    chunks = list(chunks)
    chunks[json_index] = (JSON_CHUNK, json_bytes)

    body = b"".join(
        GLB_CHUNK_HEADER.pack(len(data), chunk_type) + data
        for chunk_type, data in chunks
    )
    output = GLB_HEADER.pack(b"glTF", 2, GLB_HEADER.size + len(body)) + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def validate_hand_only(path: Path) -> dict:
    _, _, document = read_glb(path)
    names = [
        str(item.get("name", ""))
        for collection in ("nodes", "meshes")
        for item in document.get(collection, [])
    ]
    object_names = [name for name in names if name.startswith("OBJECT__")]
    if object_names:
        raise ValueError(f"Object geometry remains in {path}: {object_names}")
    external_uris = [
        item["uri"]
        for collection in ("buffers", "images")
        for item in document.get(collection, [])
        if "uri" in item
    ]
    if external_uris:
        raise ValueError(f"External URI remains in {path}: {external_uris}")
    return {
        "node_count": len(document.get("nodes", [])),
        "mesh_count": len(document.get("meshes", [])),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()

    samples = []
    for hand in ("inspire", "xhand", "leap"):
        for source in sorted((bundle_root / hand).glob("frame_*/models/prediction.glb")):
            frame_dir = source.parents[1]
            samples.append({
                "hand": hand,
                "frame": int(frame_dir.name.split("_", 1)[1]),
                "sample_dir": frame_dir.relative_to(bundle_root).as_posix(),
            })

    records = []
    for sample in samples:
        models = bundle_root / sample["sample_dir"] / "models"
        source = models / "prediction.glb"
        output = models / "prediction_hand_only.glb"
        chunks, json_index, document = read_glb(source)
        removed_nodes, removed_meshes = strip_recorded_objects(document)
        write_glb(output, chunks, json_index, document)
        validation = validate_hand_only(output)
        records.append({
            "hand": sample["hand"],
            "frame": sample["frame"],
            "source": source.relative_to(bundle_root).as_posix(),
            "output": output.relative_to(bundle_root).as_posix(),
            "removed_object_nodes": removed_nodes,
            "removed_object_meshes": removed_meshes,
            **validation,
        })

    print(json.dumps({"samples": records}, indent=2))


if __name__ == "__main__":
    main()
