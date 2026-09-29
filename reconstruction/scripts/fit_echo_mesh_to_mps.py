#!/usr/bin/env python3
"""Align ECHO's metric canonical object mesh to the Room4 MPS chair anchor."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from camera_motion_geometry import project_points
from fit_mesh_to_mps import bilinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--landmarks', type=Path, required=True)
    parser.add_argument('--sam-anchor', type=Path, required=True)
    parser.add_argument('--echo-mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    camera = np.load(args.sequence / 'aria_camera_trajectory.npz')
    anchor = np.load(args.sam_anchor)
    landmarks = np.load(args.landmarks)
    points = landmarks['clean_points_world']
    points = points[np.linspace(0, len(points)-1, min(1600, len(points))).astype(int)]
    K = anchor['K']

    echo_mesh = trimesh.load(args.echo_mesh, force='mesh', process=False)
    vertices = np.asarray(echo_mesh.vertices, dtype=np.float64)
    faces = np.asarray(echo_mesh.faces, dtype=np.int32)
    sampled = echo_mesh.sample(min(2600, len(vertices)))
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces),
    ))

    # ECHO: x=right, y=front, z=up. SAM metric: x=right, y=up,
    # z=back. This proper rotation supplies a semantic initialization; metric
    # MPS points and silhouettes estimate the final rigid alignment.
    R_sam_echo = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    sam_mesh = trimesh.load(str(anchor['fixed_mesh_path']), force='mesh', process=False)
    t_sam_echo = np.asarray(sam_mesh.vertices).mean(0) - vertices.mean(0) @ R_sam_echo.T
    T_sam_echo_initial = np.eye(4)
    T_sam_echo_initial[:3, :3] = R_sam_echo
    T_sam_echo_initial[:3, 3] = t_sam_echo
    T_world_sam = anchor['T_world_object_reference']
    T_initial = T_world_sam @ T_sam_echo_initial
    initial = np.r_[Rotation.from_matrix(T_initial[:3, :3]).as_rotvec(), T_initial[:3, 3]]

    observations = []
    for i in [20, 28, 36]:
        mask = cv2.imread(str(args.sequence / f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'), 0) > 0
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = np.concatenate(contours).reshape(-1, 2)
        contour = contour[np.linspace(0, len(contour)-1, min(180, len(contour))).astype(int)]
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        observations.append((i, np.linalg.inv(camera['T_world_camera'][i]), outside, contour))

    def unpack(parameters):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        T[:3, 3] = parameters[3:]
        return T

    def residual(parameters):
        T = unpack(parameters)
        R, t = T[:3, :3], T[:3, 3]
        local = (points - t) @ R
        distance = scene.compute_distance(o3d.core.Tensor(local.astype(np.float32)), nthreads=2).numpy()
        result = [distance / .015]
        verts_world = sampled @ R.T + t
        for _, T_camera_world, outside, contour in observations:
            uv, depth = project_points(verts_world, T_camera_world, K)
            visible = depth > .05
            result.append(bilinear(outside, uv[visible]) / 5.)
            result.append(cKDTree(uv[visible]).query(contour)[0] / 5.)
        delta_R = Rotation.from_matrix(T_initial[:3, :3].T @ R).as_rotvec()
        result.extend([delta_R / .35, (t - T_initial[:3, 3]) / .12])
        return np.concatenate(result)

    fit = least_squares(residual, initial, loss='soft_l1', f_scale=1., diff_step=1e-4,
                        max_nfev=180, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    T_world_echo = unpack(fit.x)
    T_sam_echo = np.linalg.inv(T_world_sam) @ T_world_echo
    local = (points - T_world_echo[:3, 3]) @ T_world_echo[:3, :3]
    distances = scene.compute_distance(o3d.core.Tensor(local.astype(np.float32)), nthreads=2).numpy()

    checks = []
    for i in [20, 24, 28, 32, 36, 40, 44, 48, 52, 55]:
        target = cv2.imread(str(args.sequence / f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'), 0) > 0
        T_camera_echo = np.linalg.inv(camera['T_world_camera'][i]) @ T_world_echo
        uv, depth = project_points(vertices, T_camera_echo, K)
        rendered = np.zeros(target.shape, np.uint8)
        for face in faces:
            if (depth[face] > .05).all():
                cv2.fillConvexPoly(rendered, np.rint(uv[face]).astype(np.int32), 1)
        pred = rendered > 0
        score = float((pred & target).sum() / max((pred | target).sum(), 1))
        checks.append({'frame': i, 'silhouette_iou': score, 'used_in_fit': i in [20, 28, 36]})

    np.savez_compressed(
        args.output,
        T_world_echo_reference=T_world_echo,
        T_world_sam_reference=T_world_sam,
        T_sam_echo=T_sam_echo,
        reference_frame=28,
        K=K,
        echo_mesh_path=str(args.echo_mesh.resolve()),
        translation_units='meters',
        transform_convention='column vectors: T_world_echo = T_world_sam @ T_sam_echo',
    )
    report = {
        'optimizer_success': bool(fit.success),
        'optimizer_message': fit.message,
        'surface_distance_median_m': float(np.median(distances)),
        'surface_distance_p90_m': float(np.quantile(distances, .9)),
        'held_out_silhouette_iou_median': float(np.median([x['silhouette_iou'] for x in checks if not x['used_in_fit']])),
        'silhouette_checks': checks,
        'T_sam_echo': T_sam_echo.tolist(),
        'notes': [
            'ECHO canonical whitechair scale is preserved in meters.',
            'Only the object-coordinate change is estimated here; temporal motion remains the measured SAM/MPS trajectory.',
        ],
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
