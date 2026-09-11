#!/usr/bin/env python3
"""Export a posed Inspire URDF and recorded occluder object as a material GLB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation


WORLD_Z_UP_TO_GLTF_Y_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--frame-file-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transparent-preview", type=Path, required=True)
    parser.add_argument("--alignment-check", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument(
        "--opengl-platform", choices=("egl", "osmesa", "pyglet"), default="egl"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def load_frame_inputs(args: argparse.Namespace) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    for path in (args.urdf, args.metadata, args.rgb, args.depth):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth_mm is None:
        raise ValueError("OpenCV could not read RGB or depth input")
    if depth_mm.ndim != 2 or rgb_bgr.shape[:2] != depth_mm.shape:
        raise ValueError(
            f"RGB/depth shape mismatch: {rgb_bgr.shape} versus {depth_mm.shape}"
        )
    camera = next(
        item
        for item in metadata["cameras"]
        if int(item["camera_id"]) == args.camera_id
    )
    return metadata, rgb_bgr, depth_mm, camera


def configure_urdf(robot: yourdfpy.URDF, metadata: dict) -> dict[str, float]:
    joint_state = metadata["hand"]["joint_state"]
    recorded = {
        str(name): float(value)
        for name, value in zip(joint_state["names"], joint_state["positions"])
    }
    missing = [name for name in robot.actuated_joint_names if name not in recorded]
    if missing:
        raise ValueError(f"Recorded joint state misses URDF joints: {missing}")
    configuration = {name: recorded[name] for name in robot.actuated_joint_names}
    robot.update_cfg(configuration)
    return configuration


def base_pose_world(metadata: dict) -> np.ndarray:
    pose = metadata["hand"]["base_pose_worldframe"]
    return rigid_transform(pose["rotation_matrix"], pose["position"])


def validate_fk(
    robot: yourdfpy.URDF, metadata: dict, world_from_base: np.ndarray
) -> dict:
    errors = {}
    for link_name, item in metadata["links"].items():
        if link_name not in robot.link_map:
            continue
        base_from_link = robot.get_transform(link_name)
        predicted = (world_from_base @ base_from_link)[:3, 3]
        recorded = np.asarray(item["location_worldframe"], dtype=np.float64)
        errors[link_name] = float(np.linalg.norm(predicted - recorded))
    values = np.asarray(list(errors.values()), dtype=np.float64)
    if len(values) == 0:
        raise ValueError("No recorded links matched the URDF")
    return {
        "matched_links": int(len(values)),
        "mean_error_mm": float(values.mean() * 1000.0),
        "median_error_mm": float(np.median(values) * 1000.0),
        "max_error_mm": float(values.max() * 1000.0),
        "per_link_error_mm": {
            name: float(error * 1000.0) for name, error in errors.items()
        },
    }


def add_posed_hand(
    output_scene: trimesh.Scene,
    robot: yourdfpy.URDF,
    world_from_base: np.ndarray,
) -> dict:
    material_names = []
    added = 0
    for index, node_name in enumerate(sorted(robot.scene.graph.nodes_geometry)):
        base_from_geometry, geometry_name = robot.scene.graph.get(
            frame_to=node_name, frame_from=robot.scene.graph.base_frame
        )
        geometry = robot.scene.geometry[geometry_name].copy()
        material = getattr(geometry.visual, "material", None)
        material_name = getattr(material, "name", None)
        material_names.append(str(material_name or "unnamed"))
        name = f"HAND__{index:02d}__{node_name}__{material_name or 'material'}"
        output_scene.add_geometry(
            geometry,
            geom_name=name,
            node_name=name,
            transform=(
                WORLD_Z_UP_TO_GLTF_Y_UP @ world_from_base @ base_from_geometry
            ),
        )
        added += 1
    if added == 0:
        raise ValueError("URDF did not provide any visual geometry")
    return {
        "visual_primitives": added,
        "material_names": sorted(set(material_names)),
    }


def recorded_objects(metadata: dict) -> list[dict]:
    return (
        metadata.get("domain_randomization", {})
        .get("scene", {})
        .get("occluders", {})
        .get("objects", [])
    )


def add_recorded_objects(output_scene: trimesh.Scene, metadata: dict) -> list[dict]:
    object_records = []
    for index, item in enumerate(recorded_objects(metadata)):
        size = np.asarray(item["size_xyz"], dtype=np.float64)
        position = np.asarray(item["position_worldframe"], dtype=np.float64)
        rotation_degrees = np.asarray(item["rotation_xyz_degrees"], dtype=np.float64)
        world_from_object = rigid_transform(
            Rotation.from_euler("xyz", rotation_degrees, degrees=True).as_matrix(),
            position,
        )
        diffuse = np.asarray(item["diffuse_color"], dtype=np.float64)
        rgba = np.rint(np.r_[np.clip(diffuse, 0.0, 1.0), 1.0] * 255.0).astype(
            np.uint8
        )
        material = trimesh.visual.material.PBRMaterial(
            name=f"recorded_occluder_{index}_material",
            baseColorFactor=rgba,
            metallicFactor=float(item.get("metallic", 0.0)),
            roughnessFactor=float(item.get("roughness", 1.0)),
            doubleSided=False,
        )
        box = trimesh.creation.box(extents=size)
        box.visual = trimesh.visual.TextureVisuals(material=material)
        name = f"OBJECT__{index:02d}__foreground_occluder_block"
        output_scene.add_geometry(
            box,
            geom_name=name,
            node_name=name,
            transform=WORLD_Z_UP_TO_GLTF_Y_UP @ world_from_object,
        )
        object_records.append(
            {
                "name": name,
                "path": item.get("path"),
                "size_xyz_m": size.tolist(),
                "position_worldframe_m": position.tolist(),
                "rotation_xyz_degrees": rotation_degrees.tolist(),
                "diffuse_color": diffuse.tolist(),
                "roughness": float(item.get("roughness", 1.0)),
                "metallic": float(item.get("metallic", 0.0)),
            }
        )
    return object_records


def export_scene(scene: trimesh.Scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    output.write_bytes(payload)


def add_lights(pyrender_scene, camera_pose: np.ndarray) -> None:
    import pyrender

    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    pyrender_scene.add(key, pose=camera_pose, name="camera_key_light")

    fill_pose = camera_pose.copy()
    fill_pose[:3, :3] = camera_pose[:3, :3] @ Rotation.from_euler(
        "xy", [-28.0, 32.0], degrees=True
    ).as_matrix()
    fill = pyrender.DirectionalLight(
        color=np.array([0.90, 0.94, 1.0]), intensity=1.3
    )
    pyrender_scene.add(fill, pose=fill_pose, name="camera_fill_light")


def downsample_rgba(
    rgba: np.ndarray, depth: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    premultiplied = rgba[..., :3].astype(np.float32) * alpha
    premultiplied = cv2.resize(
        premultiplied, (width, height), interpolation=cv2.INTER_AREA
    )
    alpha_small = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    if alpha_small.ndim == 2:
        alpha_small = alpha_small[..., None]
    rgb = np.divide(
        premultiplied,
        np.maximum(alpha_small, 1e-6),
        out=np.zeros_like(premultiplied),
        where=alpha_small > 1e-6,
    )
    output = np.dstack(
        (
            np.clip(np.rint(rgb), 0, 255).astype(np.uint8),
            np.clip(np.rint(alpha_small[..., 0] * 255.0), 0, 255).astype(np.uint8),
        )
    )
    depth_small = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return output, depth_small


def render_camera_view(
    scene: trimesh.Scene,
    camera: dict,
    width: int,
    height: int,
    supersample: int,
) -> tuple[np.ndarray, np.ndarray]:
    import pyrender

    render_width = width * supersample
    render_height = height * supersample
    intrinsic = np.asarray(camera["intrinsic_matrix_output"], dtype=np.float64).copy()
    intrinsic[0, :] *= supersample
    intrinsic[1, :] *= supersample
    camera_model = pyrender.IntrinsicsCamera(
        fx=float(intrinsic[0, 0]),
        fy=float(intrinsic[1, 1]),
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        znear=0.05,
        zfar=5.0,
        name="dataset_camera_0",
    )
    view = np.asarray(camera["view_matrix_world_to_camera"], dtype=np.float64)
    camera_pose = WORLD_Z_UP_TO_GLTF_Y_UP @ np.linalg.inv(view)
    render_scene = pyrender.Scene.from_trimesh_scene(
        scene,
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.32, 0.32, 0.32]),
    )
    render_scene.add(camera_model, pose=camera_pose, name="dataset_camera_0")
    add_lights(render_scene, camera_pose)
    renderer = pyrender.OffscreenRenderer(render_width, render_height)
    try:
        rgba, depth = renderer.render(
            render_scene,
            flags=(pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES),
        )
    finally:
        renderer.delete()
    rgba = np.asarray(rgba, dtype=np.uint8).copy()
    depth = np.asarray(depth, dtype=np.float32)
    alpha = np.where(depth > 0.0, 255, 0).astype(np.uint8)
    rgba[..., 3] = alpha
    rgba[alpha == 0, :3] = 0
    if supersample > 1:
        rgba, depth = downsample_rgba(rgba, depth, width, height)
    return rgba, depth


def write_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgba[..., [2, 1, 0, 3]]):
        raise OSError(f"Failed to write transparent preview: {path}")


def mask_and_depth_metrics(
    render_rgba: np.ndarray, render_depth_m: np.ndarray, depth_mm: np.ndarray
) -> dict:
    recorded_depth_m = depth_mm.astype(np.float32) * 0.001
    recorded_mask = (recorded_depth_m > 0.05) & (recorded_depth_m < 1.0)
    render_mask = render_rgba[..., 3] > 8
    intersection = recorded_mask & render_mask
    union = recorded_mask | render_mask
    iou = float(intersection.sum() / max(int(union.sum()), 1))
    precision = float(intersection.sum() / max(int(render_mask.sum()), 1))
    recall = float(intersection.sum() / max(int(recorded_mask.sum()), 1))
    valid_depth = intersection & (render_depth_m > 0.0)
    errors_mm = (
        np.abs(render_depth_m[valid_depth] - recorded_depth_m[valid_depth]) * 1000.0
    )
    return {
        "recorded_foreground_pixels": int(recorded_mask.sum()),
        "render_foreground_pixels": int(render_mask.sum()),
        "intersection_pixels": int(intersection.sum()),
        "mask_iou": iou,
        "mask_precision": precision,
        "mask_recall": recall,
        "depth_mae_mm_on_intersection": (
            None if len(errors_mm) == 0 else float(errors_mm.mean())
        ),
        "depth_median_abs_error_mm_on_intersection": (
            None if len(errors_mm) == 0 else float(np.median(errors_mm))
        ),
        "depth_p95_abs_error_mm_on_intersection": (
            None if len(errors_mm) == 0 else float(np.percentile(errors_mm, 95.0))
        ),
    }


def write_alignment_check(
    path: Path, rgb_bgr: np.ndarray, rgba: np.ndarray, metrics: dict
) -> None:
    source = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    overlay = source * (1.0 - 0.88 * alpha) + rgba[..., :3].astype(np.float32) * (
        0.88 * alpha
    )
    overlay = np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
    panel = np.hstack((source.astype(np.uint8), overlay))
    panel_bgr = panel[..., ::-1].copy()
    cv2.putText(
        panel_bgr,
        f"alignment IoU={metrics['mask_iou']:.3f}",
        (500, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), panel_bgr):
        raise OSError(f"Failed to write alignment check: {path}")


def inspect_materials(scene: trimesh.Scene) -> dict:
    material_names = []
    visual_kinds = []
    texture_images = 0
    for geometry in scene.geometry.values():
        visual_kinds.append(str(geometry.visual.kind))
        material = getattr(geometry.visual, "material", None)
        material_names.append(str(getattr(material, "name", None) or "unnamed"))
        if getattr(material, "baseColorTexture", None) is not None:
            texture_images += 1
    return {
        "geometry_count": len(scene.geometry),
        "material_names": sorted(set(material_names)),
        "visual_kinds": sorted(set(visual_kinds)),
        "primitives_with_base_color_texture": texture_images,
    }


def main() -> None:
    args = parse_args()
    if args.supersample not in (1, 2, 3, 4):
        raise ValueError("--supersample must be one of 1, 2, 3, 4")
    os.environ["PYOPENGL_PLATFORM"] = args.opengl_platform
    metadata, rgb_bgr, depth_mm, camera = load_frame_inputs(args)

    robot = yourdfpy.URDF.load(
        str(args.urdf),
        build_scene_graph=True,
        build_collision_scene_graph=False,
        load_meshes=True,
        load_collision_meshes=False,
        force_mesh=False,
    )
    configuration = configure_urdf(robot, metadata)
    world_from_base = base_pose_world(metadata)
    fk_validation = validate_fk(robot, metadata, world_from_base)

    output_scene = trimesh.Scene(base_frame="GLTF_WORLD_Y_UP")
    hand_record = add_posed_hand(output_scene, robot, world_from_base)
    object_records = add_recorded_objects(output_scene, metadata)
    if not object_records:
        raise ValueError("Frame metadata does not contain a recorded object")
    export_scene(output_scene, args.output)

    height, width = rgb_bgr.shape[:2]
    rgba, rendered_depth = render_camera_view(
        output_scene, camera, width, height, args.supersample
    )
    write_rgba(args.transparent_preview, rgba)
    alignment = mask_and_depth_metrics(rgba, rendered_depth, depth_mm)
    if args.alignment_check is not None:
        write_alignment_check(args.alignment_check, rgb_bgr, rgba, alignment)

    reloaded = trimesh.load(args.output, force="scene", process=False)
    if not isinstance(reloaded, trimesh.Scene):
        raise ValueError("Reloaded GLB is not a trimesh Scene")
    if not all(np.isfinite(geometry.vertices).all() for geometry in reloaded.geometry.values()):
        raise ValueError("Reloaded GLB contains non-finite vertices")

    manifest = {
        "artifact": str(args.output),
        "artifact_type": "posed Inspire URDF visual GLBs plus recorded object",
        "background": "none; GLB contains only geometry and PNG alpha is transparent",
        "inputs": {
            "urdf": str(args.urdf),
            "metadata": str(args.metadata),
            "rgb": str(args.rgb),
            "depth": str(args.depth),
            "urdf_sha256": sha256(args.urdf),
            "metadata_sha256": sha256(args.metadata),
            "rgb_sha256": sha256(args.rgb),
            "depth_sha256": sha256(args.depth),
        },
        "frame_file_index": args.frame_file_index,
        "metadata_frame": metadata.get("frame"),
        "camera_id": args.camera_id,
        "hand": {
            "name": metadata["hand"].get("name"),
            "base_pose_worldframe": metadata["hand"]["base_pose_worldframe"],
            "joint_configuration_radians": configuration,
            **hand_record,
        },
        "objects": object_records,
        "coordinate_system": {
            "units": "meters",
            "dataset_world": "right-handed Z-up",
            "gltf_world": "right-handed Y-up",
            "mapping": "[x, y, z] -> [x, z, -y]",
        },
        "fk_validation": fk_validation,
        "camera_alignment_validation": alignment,
        "exported_scene": inspect_materials(reloaded),
        "scene_bounds_gltf_m": reloaded.bounds.tolist(),
        "scene_extents_m": reloaded.extents.tolist(),
        "glb_bytes": args.output.stat().st_size,
        "glb_sha256": sha256(args.output),
        "transparent_preview": str(args.transparent_preview),
        "transparent_preview_sha256": sha256(args.transparent_preview),
        "transparent_preview_shape_hwc": list(rgba.shape),
        "transparent_pixels": int(np.count_nonzero(rgba[..., 3] == 0)),
        "nontransparent_pixels": int(np.count_nonzero(rgba[..., 3] > 0)),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
