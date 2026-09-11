#!/usr/bin/env python3
"""Export one bundled sample.

Portable command (run from any directory):
    python scripts/export_gt_pred_glb.py --bundle-root BUNDLE --hand inspire \
      --frame 105 --camera-id 1 --urdf BUNDLE/assets/inspire/hand_visual.urdf \
      --metadata BUNDLE/inspire/frame_000105/source/scene_gt.json \
      --prediction BUNDLE/inspire/frame_000105/source/prediction.json \
      --rgb BUNDLE/inspire/frame_000105/source/rgb.png \
      --depth BUNDLE/inspire/frame_000105/source/depth.png \
      --output-dir BUNDLE/inspire/frame_000105
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation


WORLD_Z_UP_TO_GLTF_Y_UP = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
)

PROFILES = {
    "inspire": {
        "base": "base",
        "keypoints": (
            "base", "index_intermediate", "index_proximal", "index_tip",
            "middle_intermediate", "middle_proximal", "middle_tip",
            "pinky_intermediate", "pinky_proximal", "pinky_tip",
            "ring_intermediate", "ring_proximal", "ring_tip",
            "thumb_distal", "thumb_intermediate", "thumb_proximal",
            "thumb_proximal_base", "thumb_tip",
        ),
        "skeleton": (
            ("base", "thumb_proximal_base"),
            ("thumb_proximal_base", "thumb_proximal"),
            ("thumb_proximal", "thumb_intermediate"),
            ("thumb_intermediate", "thumb_distal"),
            ("thumb_distal", "thumb_tip"),
            ("base", "index_proximal"),
            ("index_proximal", "index_intermediate"),
            ("index_intermediate", "index_tip"),
            ("base", "middle_proximal"),
            ("middle_proximal", "middle_intermediate"),
            ("middle_intermediate", "middle_tip"),
            ("base", "ring_proximal"),
            ("ring_proximal", "ring_intermediate"),
            ("ring_intermediate", "ring_tip"),
            ("base", "pinky_proximal"),
            ("pinky_proximal", "pinky_intermediate"),
            ("pinky_intermediate", "pinky_tip"),
        ),
        "q_names": None,
        "q_size": 11,
        "urdf_indices_by_q": (2, 3, 5, 4, 0, 1),
    },
    "xhand": {
        "base": "right_hand_link",
        "keypoints": (
            "right_hand_index_bend_link", "right_hand_index_rota_link1",
            "right_hand_index_rota_link2", "right_hand_index_rota_tip",
            "right_hand_link", "right_hand_mid_link1", "right_hand_mid_link2",
            "right_hand_mid_tip", "right_hand_pinky_link1",
            "right_hand_pinky_link2", "right_hand_pinky_tip",
            "right_hand_ring_link1", "right_hand_ring_link2",
            "right_hand_ring_tip", "right_hand_thumb_bend_link",
            "right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2",
            "right_hand_thumb_rota_tip",
        ),
        "skeleton": (
            ("right_hand_link", "right_hand_thumb_bend_link"),
            ("right_hand_thumb_bend_link", "right_hand_thumb_rota_link1"),
            ("right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2"),
            ("right_hand_thumb_rota_link2", "right_hand_thumb_rota_tip"),
            ("right_hand_link", "right_hand_index_bend_link"),
            ("right_hand_index_bend_link", "right_hand_index_rota_link1"),
            ("right_hand_index_rota_link1", "right_hand_index_rota_link2"),
            ("right_hand_index_rota_link2", "right_hand_index_rota_tip"),
            ("right_hand_link", "right_hand_mid_link1"),
            ("right_hand_mid_link1", "right_hand_mid_link2"),
            ("right_hand_mid_link2", "right_hand_mid_tip"),
            ("right_hand_link", "right_hand_ring_link1"),
            ("right_hand_ring_link1", "right_hand_ring_link2"),
            ("right_hand_ring_link2", "right_hand_ring_tip"),
            ("right_hand_link", "right_hand_pinky_link1"),
            ("right_hand_pinky_link1", "right_hand_pinky_link2"),
            ("right_hand_pinky_link2", "right_hand_pinky_tip"),
        ),
        "q_names": (
            "right_hand_index_bend_joint", "right_hand_mid_joint1",
            "right_hand_pinky_joint1", "right_hand_ring_joint1",
            "right_hand_thumb_bend_joint", "right_hand_index_joint1",
            "right_hand_mid_joint2", "right_hand_pinky_joint2",
            "right_hand_ring_joint2", "right_hand_thumb_rota_joint1",
            "right_hand_index_joint2", "right_hand_mid_joint3",
            "right_hand_pinky_joint3", "right_hand_ring_joint3",
            "right_hand_thumb_rota_joint2", "right_hand_index_rota_joint3",
            "right_hand_thumb_rota_joint3",
        ),
    },
    "leap": {
        "base": "base",
        "keypoints": (
            "base", "dip", "dip_2", "dip_3", "fingertip", "fingertip_2",
            "fingertip_3", "index_tip_head", "mcp_joint", "mcp_joint_2",
            "mcp_joint_3", "middle_tip_head", "pip", "pip_2", "pip_3",
            "ring_tip_head", "thumb_dip", "thumb_fingertip", "thumb_pip",
            "thumb_temp_base", "thumb_tip_head",
        ),
        "skeleton": (
            ("base", "mcp_joint"), ("mcp_joint", "pip"),
            ("pip", "dip"), ("dip", "fingertip"),
            ("fingertip", "index_tip_head"),
            ("base", "mcp_joint_2"), ("mcp_joint_2", "pip_2"),
            ("pip_2", "dip_2"), ("dip_2", "fingertip_2"),
            ("fingertip_2", "middle_tip_head"),
            ("base", "mcp_joint_3"), ("mcp_joint_3", "pip_3"),
            ("pip_3", "dip_3"), ("dip_3", "fingertip_3"),
            ("fingertip_3", "ring_tip_head"),
            ("base", "thumb_temp_base"),
            ("thumb_temp_base", "thumb_pip"),
            ("thumb_pip", "thumb_dip"),
            ("thumb_dip", "thumb_fingertip"),
            ("thumb_fingertip", "thumb_tip_head"),
        ),
        "q_names": tuple(str(index) for index in range(16)),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--hand", choices=tuple(PROFILES), required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--opengl-platform", choices=("egl", "osmesa", "pyglet"), default="egl")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    value[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return value


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


def load_robot(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(path), build_scene_graph=True, build_collision_scene_graph=False,
        load_meshes=True, load_collision_meshes=False, force_mesh=False,
    )


def gt_configuration(robot: yourdfpy.URDF, metadata: dict) -> dict[str, float]:
    state = metadata["hand"]["joint_state"]
    recorded = dict(zip(state["names"], state["positions"]))
    result = {}
    for name in robot.actuated_joint_names:
        candidates = (name, f"a_{name}") if name.isdigit() else (name,)
        source = next((candidate for candidate in candidates if candidate in recorded), None)
        if source is None:
            raise KeyError(f"GT joint {name!r} is absent from bundled metadata")
        result[name] = float(recorded[source])
    return result


def prediction_configuration(robot: yourdfpy.URDF, prediction: dict, profile: dict) -> dict[str, float]:
    q = np.asarray(prediction["q_independent"], dtype=np.float64)
    if profile.get("urdf_indices_by_q") is not None:
        expanded = np.zeros(int(profile["q_size"]), dtype=np.float64)
        for q_index, urdf_index in enumerate(profile["urdf_indices_by_q"]):
            expanded[urdf_index] = q[q_index]
        return dict(zip(robot.actuated_joint_names, expanded.tolist()))
    names = tuple(profile["q_names"])
    if len(q) != len(names):
        raise ValueError(f"Prediction q has {len(q)} entries; expected {len(names)}")
    return dict(zip(names, q.tolist()))


def keypoint_positions(robot: yourdfpy.URDF, profile: dict) -> np.ndarray:
    base = str(profile["base"])
    return np.stack([
        robot.get_transform(name, frame_from=base)[:3, 3]
        for name in profile["keypoints"]
    ])


def metadata_keypoints(metadata: dict, profile: dict) -> np.ndarray:
    return np.stack([
        np.asarray(metadata["links"][name]["location_worldframe"], dtype=np.float64)
        for name in profile["keypoints"]
    ])


def recorded_objects(metadata: dict) -> list[dict]:
    return metadata.get("domain_randomization", {}).get("scene", {}).get(
        "occluders", {}).get("objects", [])


def add_posed_hand(scene: trimesh.Scene, robot: yourdfpy.URDF,
                    world_from_base: np.ndarray, profile: dict) -> dict:
    materials = []
    nodes = sorted(robot.scene.graph.nodes_geometry)
    for index, node_name in enumerate(nodes):
        base_from_geometry, geometry_name = robot.scene.graph.get(
            frame_to=node_name, frame_from=str(profile["base"])
        )
        geometry = robot.scene.geometry[geometry_name].copy()
        material = getattr(geometry.visual, "material", None)
        material_name = getattr(material, "name", None) or "material"
        materials.append(str(material_name))
        name = f"HAND__{index:03d}__{safe_name(node_name)}__{safe_name(material_name)}"
        scene.add_geometry(
            geometry, geom_name=name, node_name=name,
            transform=WORLD_Z_UP_TO_GLTF_Y_UP @ world_from_base @ base_from_geometry,
        )
    if not nodes:
        raise ValueError("URDF contains no visual geometry")
    return {"visual_primitive_count": len(nodes), "material_names": sorted(set(materials))}


def add_recorded_objects(scene: trimesh.Scene, metadata: dict) -> list[dict]:
    records = []
    for index, item in enumerate(recorded_objects(metadata)):
        size = np.asarray(item["size_xyz"], dtype=np.float64)
        position = np.asarray(item["position_worldframe"], dtype=np.float64)
        rotation_deg = np.asarray(item["rotation_xyz_degrees"], dtype=np.float64)
        world_from_object = rigid_transform(
            Rotation.from_euler("xyz", rotation_deg, degrees=True).as_matrix(), position
        )
        diffuse = np.asarray(item["diffuse_color"], dtype=np.float64)
        rgba = np.rint(np.r_[np.clip(diffuse, 0.0, 1.0), 1.0] * 255.0).astype(np.uint8)
        material = trimesh.visual.material.PBRMaterial(
            name=f"recorded_object_{index}_material", baseColorFactor=rgba,
            metallicFactor=float(item.get("metallic", 0.0)),
            roughnessFactor=float(item.get("roughness", 1.0)), doubleSided=False,
        )
        box = trimesh.creation.box(extents=size)
        box.visual = trimesh.visual.TextureVisuals(material=material)
        name = f"OBJECT__{index:02d}__recorded_box"
        scene.add_geometry(box, geom_name=name, node_name=name,
                           transform=WORLD_Z_UP_TO_GLTF_Y_UP @ world_from_object)
        records.append({
            "name": name,
            "scene_prim": str(item.get("scene_prim", item.get("path", ""))).lstrip("/"),
            "size_xyz_m": size.tolist(), "position_worldframe_m": position.tolist(),
            "rotation_xyz_degrees": rotation_deg.tolist(),
            "diffuse_color": diffuse.tolist(),
            "roughness": float(item.get("roughness", 1.0)),
            "metallic": float(item.get("metallic", 0.0)),
        })
    return records


def build_scene(robot: yourdfpy.URDF, world_from_base: np.ndarray,
                metadata: dict, profile: dict) -> tuple[trimesh.Scene, dict, list[dict]]:
    scene = trimesh.Scene(base_frame="GLTF_WORLD_Y_UP")
    hand_record = add_posed_hand(scene, robot, world_from_base, profile)
    objects = add_recorded_objects(scene, metadata)
    return scene, hand_record, objects


def export_scene(scene: trimesh.Scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))


def add_lights(scene, camera_pose: np.ndarray) -> None:
    import pyrender

    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        pose=camera_pose, name="camera_key_light",
    )
    fill_pose = camera_pose.copy()
    fill_pose[:3, :3] = camera_pose[:3, :3] @ Rotation.from_euler(
        "xy", [-28.0, 32.0], degrees=True
    ).as_matrix()
    scene.add(
        pyrender.DirectionalLight(color=np.array([0.90, 0.94, 1.0]), intensity=1.3),
        pose=fill_pose, name="camera_fill_light",
    )


def downsample_rgba(rgba: np.ndarray, depth: np.ndarray,
                    width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    premultiplied = rgba[..., :3].astype(np.float32) * alpha
    premultiplied = cv2.resize(premultiplied, (width, height), interpolation=cv2.INTER_AREA)
    alpha_small = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    if alpha_small.ndim == 2:
        alpha_small = alpha_small[..., None]
    rgb = np.divide(
        premultiplied, np.maximum(alpha_small, 1e-6),
        out=np.zeros_like(premultiplied), where=alpha_small > 1e-6,
    )
    output = np.dstack((
        np.clip(np.rint(rgb), 0, 255).astype(np.uint8),
        np.clip(np.rint(alpha_small[..., 0] * 255.0), 0, 255).astype(np.uint8),
    ))
    return output, cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


def render_camera_view(scene: trimesh.Scene, camera: dict, width: int,
                       height: int, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    import pyrender

    render_width, render_height = width * supersample, height * supersample
    intrinsic = np.asarray(camera["intrinsic_matrix_output"], dtype=np.float64).copy()
    intrinsic[0, :] *= supersample
    intrinsic[1, :] *= supersample
    camera_model = pyrender.IntrinsicsCamera(
        fx=float(intrinsic[0, 0]), fy=float(intrinsic[1, 1]),
        cx=float(intrinsic[0, 2]), cy=float(intrinsic[1, 2]),
        znear=0.05, zfar=5.0, name="dataset_camera",
    )
    view = np.asarray(camera["view_matrix_world_to_camera"], dtype=np.float64)
    camera_pose = WORLD_Z_UP_TO_GLTF_Y_UP @ np.linalg.inv(view)
    render_scene = pyrender.Scene.from_trimesh_scene(
        scene, bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.32, 0.32, 0.32]),
    )
    render_scene.add(camera_model, pose=camera_pose, name="dataset_camera")
    add_lights(render_scene, camera_pose)
    renderer = pyrender.OffscreenRenderer(render_width, render_height)
    try:
        rgba, depth = renderer.render(
            render_scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
    finally:
        renderer.delete()
    rgba = np.asarray(rgba, dtype=np.uint8).copy()
    depth = np.asarray(depth, dtype=np.float32)
    rgba[..., 3] = np.where(depth > 0.0, 255, 0).astype(np.uint8)
    rgba[rgba[..., 3] == 0, :3] = 0
    if supersample > 1:
        rgba, depth = downsample_rgba(rgba, depth, width, height)
    return rgba, depth


def write_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgba[..., [2, 1, 0, 3]]):
        raise OSError(f"Could not write {path}")


def alpha_composite(rgba: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(np.rint(rgba[..., :3] * alpha + bg * (1.0 - alpha)), 0, 255).astype(np.uint8)


def finger_color(name_a: str, name_b: str = "") -> tuple[int, int, int]:
    value = f"{name_a} {name_b}".lower()
    if "thumb" in value:
        return (38, 150, 255)
    if "index" in value or value.endswith(" mcp_joint") or "fingertip index" in value:
        return (255, 205, 45)
    if "middle" in value or "_2" in value:
        return (70, 220, 110)
    if "ring" in value or "_3" in value:
        return (220, 90, 245)
    if "pinky" in value:
        return (255, 105, 105)
    return (250, 210, 80)


def prediction_2d_points(prediction: dict) -> dict[str, np.ndarray]:
    points = {}
    for item in prediction.get("keypoints", []):
        xy = item.get("pred_keypoint_2d_xy")
        if xy is None:
            xy = item.get("stage2_keypoint_xy", item.get("keypoint_xy"))
        if xy is None:
            continue
        value = np.asarray(xy, dtype=np.float64).reshape(-1)
        if len(value) >= 2 and np.isfinite(value[:2]).all() and (value[:2] > -100).all():
            points[str(item["name"])] = value[:2]
    return points


def draw_skeleton(rgb: np.ndarray, prediction: dict, profile: dict) -> tuple[np.ndarray, int]:
    panel = rgb.copy()
    points = prediction_2d_points(prediction)
    for first, second in profile["skeleton"]:
        if first not in points or second not in points:
            continue
        p1 = tuple(np.rint(points[first]).astype(int))
        p2 = tuple(np.rint(points[second]).astype(int))
        color_rgb = finger_color(first, second)
        cv2.line(panel, p1, p2, color_rgb[::-1], 4, cv2.LINE_AA)
        cv2.line(panel, p1, p2, (245, 245, 245), 1, cv2.LINE_AA)
    for name, xy in points.items():
        point = tuple(np.rint(xy).astype(int))
        color_rgb = finger_color(name)
        cv2.circle(panel, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, point, 4, color_rgb[::-1], -1, cv2.LINE_AA)
    return panel, len(points)


def make_four_column(path: Path, rgb_bgr: np.ndarray, gt_rgba: np.ndarray,
                     pred_rgba: np.ndarray, skeleton_bgr: np.ndarray,
                     hand: str, frame: int, add_mm: float) -> None:
    panel_size = (480, 480)
    original = cv2.resize(rgb_bgr, panel_size, interpolation=cv2.INTER_AREA)
    neutral = (246, 246, 246)
    gt_bgr = alpha_composite(gt_rgba, neutral)[..., ::-1]
    pred_bgr = alpha_composite(pred_rgba, neutral)[..., ::-1]
    gt_bgr = cv2.resize(gt_bgr, panel_size, interpolation=cv2.INTER_AREA)
    pred_bgr = cv2.resize(pred_bgr, panel_size, interpolation=cv2.INTER_AREA)
    skeleton = cv2.resize(skeleton_bgr, panel_size, interpolation=cv2.INTER_AREA)
    panels = (original, gt_bgr, pred_bgr, skeleton)
    titles = ("Original RGB", "GT 3D GLB", "Prediction 3D GLB", "Prediction 2D keypoints + skeleton")
    gap, header = 8, 62
    canvas = np.full((header + panel_size[1], 4 * panel_size[0] + 3 * gap, 3), 247, np.uint8)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = index * (panel_size[0] + gap)
        canvas[header:, x:x + panel_size[0]] = panel
        cv2.putText(canvas, title, (x + 10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.63, (26, 26, 26), 2, cv2.LINE_AA)
    subtitle = f"{hand.upper()}  frame {frame}  mean GT 3D error {add_mm:.3f} mm"
    cv2.putText(canvas, subtitle, (10, 53), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (82, 82, 82), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def mask_depth_metrics(rgba: np.ndarray, rendered_depth: np.ndarray,
                       depth_mm: np.ndarray) -> dict:
    recorded = depth_mm.astype(np.float32) * 0.001
    recorded_mask = (recorded > 0.05) & (recorded < 1.0)
    render_mask = rgba[..., 3] > 8
    intersection, union = recorded_mask & render_mask, recorded_mask | render_mask
    valid = intersection & (rendered_depth > 0.0)
    error = np.abs(rendered_depth[valid] - recorded[valid]) * 1000.0
    return {
        "recorded_foreground_pixels": int(recorded_mask.sum()),
        "render_foreground_pixels": int(render_mask.sum()),
        "intersection_pixels": int(intersection.sum()),
        "mask_iou": float(intersection.sum() / max(int(union.sum()), 1)),
        "mask_precision": float(intersection.sum() / max(int(render_mask.sum()), 1)),
        "mask_recall": float(intersection.sum() / max(int(recorded_mask.sum()), 1)),
        "depth_mae_mm_on_intersection": None if not len(error) else float(error.mean()),
        "depth_median_abs_error_mm_on_intersection": None if not len(error) else float(np.median(error)),
    }


def inspect_glb(path: Path) -> dict:
    raw = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total_length != len(raw):
        raise ValueError(f"Invalid GLB header: {path}")
    json_length, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"GLB does not start with JSON chunk: {path}")
    document = json.loads(raw[20:20 + json_length].decode("utf-8").rstrip(" \t\r\n\x00"))
    external_uris = []
    for collection in ("buffers", "images"):
        external_uris.extend(item["uri"] for item in document.get(collection, []) if "uri" in item)
    loaded = trimesh.load(path, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError(f"GLB did not reload as a non-empty scene: {path}")
    if not all(np.isfinite(geometry.vertices).all() for geometry in loaded.geometry.values()):
        raise ValueError(f"GLB contains non-finite vertices: {path}")
    return {
        "bytes": len(raw), "sha256": sha256(path),
        "geometry_count": len(loaded.geometry),
        "scene_extents_m": loaded.extents.tolist(),
        "external_uri_count": len(external_uris), "external_uris": external_uris,
    }


def bundle_relative(path: Path, bundle_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(bundle_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path escapes the self-contained bundle: {path}") from exc


def validate_poses(gt_robot: yourdfpy.URDF, pred_robot: yourdfpy.URDF,
                   metadata: dict, prediction: dict, camera: dict,
                   profile: dict, world_from_gt: np.ndarray,
                   camera_from_pred: np.ndarray) -> dict:
    local_gt = keypoint_positions(gt_robot, profile)
    local_pred = keypoint_positions(pred_robot, profile)
    ones = np.ones((len(local_gt), 1), dtype=np.float64)
    gt_world = (world_from_gt @ np.column_stack((local_gt, ones)).T).T[:, :3]
    recorded_world = metadata_keypoints(metadata, profile)
    view = np.asarray(camera["view_matrix_world_to_camera"], dtype=np.float64)
    recorded_camera = (
        view @ np.column_stack((recorded_world, ones)).T
    ).T[:, :3]
    predicted_camera = (
        camera_from_pred @ np.column_stack((local_pred, ones)).T
    ).T[:, :3]

    record_by_name = {str(item["name"]): item for item in prediction.get("keypoints", [])}
    stored_gt_camera = []
    for name in profile["keypoints"]:
        item = record_by_name.get(name)
        if item is None or item.get("gt_cam_xyz") is None:
            raise KeyError(f"Prediction record has no GT camera point for {name!r}")
        stored_gt_camera.append(item["gt_cam_xyz"])
    stored_gt_camera = np.asarray(stored_gt_camera, dtype=np.float64)

    fk_error = np.linalg.norm(gt_world - recorded_world, axis=1)
    report_drift = np.linalg.norm(stored_gt_camera - recorded_camera, axis=1)
    predicted_error = np.linalg.norm(predicted_camera - recorded_camera, axis=1)
    recomputed_add_m = float(predicted_error.mean())
    reported_add_m = float(prediction["summary"]["mean_gt_3d_error_m"])
    if float(fk_error.max()) > 1e-4:
        raise ValueError(f"GT FK mismatch is too large: {fk_error.max() * 1000.0:.6f} mm")
    if float(report_drift.max()) > 1e-7:
        raise ValueError(f"Report GT does not match bundled test data: {report_drift.max() * 1000.0:.6f} mm")
    if abs(recomputed_add_m - reported_add_m) > 1e-7:
        raise ValueError(
            f"Recomputed prediction error {recomputed_add_m} != report {reported_add_m}"
        )
    return {
        "keypoint_count": len(local_gt),
        "gt_fk_mean_error_mm": float(fk_error.mean() * 1000.0),
        "gt_fk_max_error_mm": float(fk_error.max() * 1000.0),
        "report_vs_current_gt_mean_drift_mm": float(report_drift.mean() * 1000.0),
        "report_vs_current_gt_max_drift_mm": float(report_drift.max() * 1000.0),
        "reported_mean_gt_3d_error_mm": reported_add_m * 1000.0,
        "recomputed_mean_gt_3d_error_mm": recomputed_add_m * 1000.0,
        "metric_delta_mm": abs(recomputed_add_m - reported_add_m) * 1000.0,
    }


def main() -> None:
    args = parse_args()
    if args.supersample not in (1, 2, 3, 4):
        raise ValueError("--supersample must be one of 1, 2, 3, 4")
    os.environ["PYOPENGL_PLATFORM"] = args.opengl_platform
    bundle_root = args.bundle_root.resolve()
    paths = (args.urdf, args.metadata, args.prediction, args.rgb, args.depth)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        bundle_relative(path, bundle_root)
    output_dir = args.output_dir.resolve()
    bundle_relative(output_dir, bundle_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth_mm is None:
        raise ValueError("OpenCV could not read bundled RGB/depth")
    if depth_mm.ndim != 2 or depth_mm.shape != rgb_bgr.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb_bgr.shape} vs {depth_mm.shape}")
    camera = next(
        item for item in metadata["cameras"]
        if int(item["camera_id"]) == int(args.camera_id)
    )
    profile = PROFILES[args.hand]

    gt_robot, pred_robot = load_robot(args.urdf), load_robot(args.urdf)
    gt_cfg = gt_configuration(gt_robot, metadata)
    pred_cfg = prediction_configuration(pred_robot, prediction, profile)
    gt_robot.update_cfg(gt_cfg)
    pred_robot.update_cfg(pred_cfg)

    base_pose = metadata["hand"]["base_pose_worldframe"]
    world_from_gt = rigid_transform(base_pose["rotation_matrix"], base_pose["position"])
    camera_from_pred = rigid_transform(
        Rotation.from_rotvec(
            np.asarray(prediction["root_rotvec_cam_from_urdf"], dtype=np.float64)
        ).as_matrix(),
        np.asarray(prediction["root_translation_cam_from_urdf"], dtype=np.float64),
    )
    view = np.asarray(camera["view_matrix_world_to_camera"], dtype=np.float64)
    world_from_pred = np.linalg.inv(view) @ camera_from_pred
    pose_validation = validate_poses(
        gt_robot, pred_robot, metadata, prediction, camera, profile,
        world_from_gt, camera_from_pred,
    )

    gt_scene, gt_hand, gt_objects = build_scene(
        gt_robot, world_from_gt, metadata, profile
    )
    pred_scene, pred_hand, pred_objects = build_scene(
        pred_robot, world_from_pred, metadata, profile
    )
    if gt_objects != pred_objects:
        raise AssertionError("GT and prediction scenes did not receive identical recorded objects")

    gt_glb = output_dir / "models" / "gt.glb"
    pred_glb = output_dir / "models" / "prediction.glb"
    gt_png = output_dir / "previews" / "gt_transparent.png"
    pred_png = output_dir / "previews" / "prediction_transparent.png"
    skeleton_png = output_dir / "previews" / "prediction_2d_skeleton.png"
    four_column_png = output_dir / "previews" / "four_column.png"
    result_json = output_dir / "metadata" / "result.json"
    export_scene(gt_scene, gt_glb)
    export_scene(pred_scene, pred_glb)

    height, width = rgb_bgr.shape[:2]
    gt_rgba, gt_render_depth = render_camera_view(
        gt_scene, camera, width, height, args.supersample
    )
    pred_rgba, _ = render_camera_view(
        pred_scene, camera, width, height, args.supersample
    )
    write_rgba(gt_png, gt_rgba)
    write_rgba(pred_png, pred_rgba)
    skeleton_bgr, skeleton_points = draw_skeleton(rgb_bgr, prediction, profile)
    skeleton_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(skeleton_png), skeleton_bgr):
        raise OSError(f"Could not write {skeleton_png}")
    make_four_column(
        four_column_png, rgb_bgr, gt_rgba, pred_rgba, skeleton_bgr,
        args.hand, args.frame,
        pose_validation["reported_mean_gt_3d_error_mm"],
    )

    gt_glb_info, pred_glb_info = inspect_glb(gt_glb), inspect_glb(pred_glb)
    if gt_glb_info["external_uri_count"] or pred_glb_info["external_uri_count"]:
        raise ValueError("Exported GLB contains an external URI")
    result = {
        "schema_version": 1,
        "hand": args.hand,
        "frame_file_index": int(args.frame),
        "metadata_frame": metadata.get("frame"),
        "camera_id": int(args.camera_id),
        "background": "none in both GLBs and transparent render previews",
        "inputs": {
            "urdf": bundle_relative(args.urdf, bundle_root),
            "metadata": bundle_relative(args.metadata, bundle_root),
            "prediction": bundle_relative(args.prediction, bundle_root),
            "rgb": bundle_relative(args.rgb, bundle_root),
            "depth": bundle_relative(args.depth, bundle_root),
            "sha256": {
                "urdf": sha256(args.urdf), "metadata": sha256(args.metadata),
                "prediction": sha256(args.prediction), "rgb": sha256(args.rgb),
                "depth": sha256(args.depth),
            },
        },
        "artifacts": {
            "gt_glb": bundle_relative(gt_glb, bundle_root),
            "prediction_glb": bundle_relative(pred_glb, bundle_root),
            "gt_transparent_preview": bundle_relative(gt_png, bundle_root),
            "prediction_transparent_preview": bundle_relative(pred_png, bundle_root),
            "prediction_2d_skeleton": bundle_relative(skeleton_png, bundle_root),
            "four_column_preview": bundle_relative(four_column_png, bundle_root),
        },
        "coordinate_system": {
            "units": "meters", "dataset_world": "right-handed Z-up",
            "gltf_world": "right-handed Y-up", "mapping": "[x,y,z] -> [x,z,-y]",
        },
        "pose_validation": pose_validation,
        "camera_alignment_validation": mask_depth_metrics(
            gt_rgba, gt_render_depth, depth_mm
        ),
        "prediction_2d_keypoints_drawn": int(skeleton_points),
        "hand_geometry": {"gt": gt_hand, "prediction": pred_hand},
        "recorded_objects_shared_by_gt_and_prediction": gt_objects,
        "glb": {"gt": gt_glb_info, "prediction": pred_glb_info},
        "preview": {
            "transparent_shape_hwc": list(gt_rgba.shape),
            "gt_nontransparent_pixels": int(np.count_nonzero(gt_rgba[..., 3] > 0)),
            "prediction_nontransparent_pixels": int(np.count_nonzero(pred_rgba[..., 3] > 0)),
            "four_column_shape_hwc": [542, 1944, 3],
        },
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "hand": args.hand, "frame": args.frame,
        "gt_3d_error_mm": pose_validation["recomputed_mean_gt_3d_error_mm"],
        "gt_glb": bundle_relative(gt_glb, bundle_root),
        "prediction_glb": bundle_relative(pred_glb, bundle_root),
        "four_column": bundle_relative(four_column_png, bundle_root),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
