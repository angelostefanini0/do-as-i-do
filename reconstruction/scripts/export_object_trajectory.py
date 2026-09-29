#!/usr/bin/env python3
"""Export camera-relative rigid poses, fixed mesh scale, and optional metric world poses.

T_camera_object acts on COLUMN vectors of the fixed, scale-baked OBJ mesh.
Missing frame poses remain NaN. No interpolation or invented timestamps.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True, help="layout_camera_frame.json")
    parser.add_argument("--timestamps-ns", type=Path, required=True, help="One device-clock timestamp per input frame")
    parser.add_argument("--object", default="whitechair")
    parser.add_argument("--mesh", type=Path, required=True, help="Original reference-frame OBJ")
    parser.add_argument("--fixed-mesh", type=Path, required=True, help="Output OBJ with the one fixed scale baked in")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-trajectory", type=Path, help="Aria camera NPZ or equivalent, OpenCV camera-to-world")
    parser.add_argument("--verified-meters-per-unit", type=float,
                        help="Supply only AFTER independently verifying the object translation and mesh scale")
    args = parser.parse_args()
    data = json.loads(args.layout.read_text())
    if data.get("frame") != "camera_frame":
        raise ValueError("Expected layout_camera_frame.json")
    timestamps = np.load(args.timestamps_ns, allow_pickle=False)
    if timestamps.ndim != 1 or timestamps.dtype.kind not in "iu" or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be a strictly increasing integer nanosecond vector")
    n = len(timestamps)
    translations = np.full((n, 3), np.nan)
    rotations = np.full((n, 3, 3), np.nan)
    quaternions = np.full((n, 4), np.nan)
    transforms = np.full((n, 4, 4), np.nan)
    valid = np.zeros(n, dtype=bool)
    fixed_scale = None
    # These are the exact column-vector boundaries of track_object.py:
    # OBJ y-up -> SAM3D z-up, then PyTorch3D -> OpenCV camera axes.
    A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    D = np.diag([-1., -1., 1.])
    unit = args.verified_meters_per_unit
    if unit is not None and (not np.isfinite(unit) or unit <= 0):
        raise ValueError("verified-meters-per-unit must be finite and positive")
    for obj in data["objects"]:
        if not obj.get("mesh_obj", "").startswith(args.object + "_frame"):
            continue
        i = int(obj["frame_idx"])
        if i < 0 or i >= n or valid[i]:
            raise ValueError(f"Duplicate or out-of-range frame {i}")
        pose = obj["local_to_scene"]
        scale = np.asarray(pose["scale"], dtype=float).reshape(-1)
        if len(scale) == 1:
            scale = np.repeat(scale, 3)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"Invalid scale at frame {i}")
        if fixed_scale is None:
            fixed_scale = scale
        elif not np.allclose(fixed_scale, scale, rtol=1e-5, atol=1e-7):
            raise ValueError("Scale changes across frames; use --fix_scale_to_init_frame")
        raw_q = np.asarray(pose["quat_wxyz"], dtype=float)
        raw_t = np.asarray(pose["translation"], dtype=float)
        if not np.isfinite(raw_q).all() or not np.isfinite(raw_t).all() or np.linalg.norm(raw_q) < 1e-12:
            continue
        # compose_transform uses x_row @ R_raw. Its column rotation is R_raw.T.
        R_raw = Rotation.from_quat(raw_q[[1, 2, 3, 0]]).as_matrix()
        rotations[i] = D @ R_raw.T @ A
        translations[i] = np.asarray(pose["translation_camera_frame"], dtype=float)
        if not np.allclose(translations[i], D @ raw_t, rtol=1e-5, atol=1e-6):
            raise ValueError("Camera conversion has a non-unit translation scale; export with matching units")
        if unit is not None:
            translations[i] *= unit
        quaternions[i] = Rotation.from_matrix(rotations[i]).as_quat()[[3, 0, 1, 2]]
        transforms[i] = np.eye(4)
        transforms[i, :3, :3] = rotations[i]
        transforms[i, :3, 3] = translations[i]
        valid[i] = True
    if not valid.any():
        raise ValueError("No valid object poses found")
    # Bake constant scale in OBJ axes, including anisotropic scale if present.
    # R_camera_object @ S_obj == D @ R_raw.T @ diag(scale) @ A.
    mesh_scale = A.T @ np.diag(fixed_scale) @ A
    if unit is not None:
        mesh_scale *= unit
    import trimesh
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    mesh.vertices = np.asarray(mesh.vertices) @ mesh_scale.T
    args.fixed_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.fixed_mesh)
    payload = dict(timestamps_ns=timestamps, timestamps=timestamps.astype(np.float64) * 1e-9,
                   relative_timestamps=(timestamps - timestamps[0]) * 1e-9,
                   frame_indices=np.arange(n), valid=valid, translations=translations,
                   rotations=rotations, quaternions_wxyz=quaternions,
                   transforms=transforms, T_camera_object=transforms,
                   fixed_scale_sam3d=fixed_scale, mesh_scale_matrix=mesh_scale,
                   fixed_mesh_path=str(args.fixed_mesh.resolve()),
                   source_mesh_path=str(args.mesh.resolve()),
                   coordinate_frame="camera: x-right,y-down,z-forward",
                   transform_convention="column vectors; camera <- fixed scale-baked OBJ",
                   timestamp_domain="Aria device clock (seconds / nanoseconds), not UTC",
                   metric_scale_verified=unit is not None,
                   translation_units="meters" if unit is not None else "unverified SAM3D/MoGe units")
    if args.camera_trajectory:
        camera = np.load(args.camera_trajectory, allow_pickle=False)
        if not np.array_equal(camera["timestamps_ns"], timestamps):
            raise ValueError("Camera and object timestamps must match exactly")
        if str(camera["translation_units"]) != "meters":
            raise ValueError("Expected camera translations in meters")
        payload["T_world_camera"] = camera["T_world_camera"]
        payload["camera_valid"] = camera["valid"]
        if unit is not None:
            world_valid = valid & camera["valid"]
            world = np.full_like(transforms, np.nan)
            world[world_valid] = camera["T_world_camera"][world_valid] @ transforms[world_valid]
            payload.update(T_world_object=world, world_valid=world_valid)
        else:
            print("Saved metric camera poses separately; world-object composition requires verified scale")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Saved {valid.sum()}/{n} valid object poses to {args.output}")
    print(f"Fixed mesh (scale baked in): {args.fixed_mesh}")


if __name__ == "__main__":
    main()
