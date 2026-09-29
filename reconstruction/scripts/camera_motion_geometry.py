"""Calibrated camera geometry for rigid-object tracking with metric MPS poses."""
import cv2
import numpy as np


def project_points(points_object, T_camera_object, intrinsics):
    points_camera = np.asarray(points_object) @ T_camera_object[:3, :3].T + T_camera_object[:3, 3]
    homogeneous = points_camera @ np.asarray(intrinsics).T
    return homogeneous[:, :2] / homogeneous[:, 2:3], points_camera[:, 2]


def propagate_pose(T_world_camera_previous, T_world_camera_current, T_camera_object_previous):
    """All translations must have the same, metric units."""
    return np.linalg.inv(T_world_camera_current) @ T_world_camera_previous @ T_camera_object_previous


def estimate_rigid_pose(points_object, pixels, intrinsics, prior=None, reprojection_threshold=3., min_inliers=8):
    """Robust PnP with a prior expressed in the CURRENT camera frame.

    Return no pose when correspondences do not support a valid estimate. The
    prior is never emitted as an observed pose on an estimation failure.
    """
    points = np.ascontiguousarray(points_object, dtype=np.float64)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1)
    points, pixels = points[finite], pixels[finite]
    if len(points) < max(6, min_inliers):
        return None, {'status': 'insufficient_correspondences', 'correspondences': len(points)}
    rvec = tvec = None
    if prior is not None:
        rvec = cv2.Rodrigues(np.ascontiguousarray(prior[:3, :3]))[0]
        tvec = np.ascontiguousarray(prior[:3, 3].reshape(3, 1))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(points, pixels, np.asarray(intrinsics, dtype=np.float64), None, rvec=rvec, tvec=tvec, useExtrinsicGuess=prior is not None, iterationsCount=300, reprojectionError=reprojection_threshold, confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, {'status': 'pnp_failed', 'correspondences': len(points), 'inliers': 0 if inliers is None else len(inliers)}
    indices = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(points[indices], pixels[indices], np.asarray(intrinsics, dtype=np.float64), None, rvec, tvec)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.reshape(3)
    projected, depth = project_points(points, transform, intrinsics)
    errors = np.linalg.norm(projected - pixels, axis=1)
    rms = float(np.sqrt(np.mean(errors[indices] ** 2)))
    ratio = len(indices) / len(points)
    support = np.linalg.svd(points[indices] - points[indices].mean(axis=0), compute_uv=False)
    info = {'status': 'ok', 'correspondences': len(points), 'inliers': len(indices), 'inlier_ratio': float(ratio), 'reprojection_rms_px': rms, 'reprojection_median_px': float(np.median(errors[indices])), 'support_singular_values_m': support.tolist()}
    if not np.isfinite(transform).all() or np.any(depth[indices] <= 0) or ratio < .5 or rms > reprojection_threshold:
        info['status'] = 'rejected_geometry'
        return None, info
    return transform, info
