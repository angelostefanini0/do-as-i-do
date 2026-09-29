#!/usr/bin/env python3
"""Select metric Aria MPS surface landmarks on one segmented object.

The interval is a verified stationary calibration interval. It is only used to
identify the object's initial 3D surface; later object motion is unconstrained.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd

from camera_motion_geometry import project_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--mps-points', type=Path, required=True)
    parser.add_argument('--object', default='whitechair')
    parser.add_argument('--reference', type=int, default=28)
    parser.add_argument('--static-frames', required=True, help='Verified stationary interval START:END')
    parser.add_argument('--sample-step', type=int, default=4)
    parser.add_argument('--num-landmarks', type=int, default=96)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start, end = map(int, args.static_frames.split(':'))
    frames = list(range(start, end, args.sample_step))
    if args.reference not in frames:
        raise ValueError('Reference must be one of the sampled static frames')
    camera = np.load(args.sequence / 'aria_camera_trajectory.npz')
    if not camera['valid'][frames].all():
        raise ValueError('All static calibration frames require valid MPS camera poses')
    meta = json.loads((args.sequence / 'aria_metadata.json').read_text())
    focal = meta['focal_px'];cx, cy = meta['principal_point_px']
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]])
    table = pd.read_csv(args.mps_points)
    quality = (table['inv_dist_std'].to_numpy() < .005) & (table['dist_std'].to_numpy() < .02)
    points = table.loc[quality, ['px_world','py_world','pz_world']].to_numpy(dtype=np.float64)
    uids = table.loc[quality, 'uid'].to_numpy()
    votes = np.zeros(len(points), np.uint8)
    ref_uv = ref_depth = None
    for i in frames:
        mask = cv2.imread(str(args.sequence / f'video_segmentation/masks/frame_{i:06d}_masks/{args.object}.png'), 0)
        if mask is None or not mask.any():
            raise ValueError(f'Missing static segmentation mask at frame {i}')
        height, width = mask.shape
        uv, depth = project_points(points, np.linalg.inv(camera['T_world_camera'][i]), K)
        x, y = np.rint(uv).astype(int).T
        inside = (depth > .1) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        hit = np.zeros(len(points), bool)
        idx = np.flatnonzero(inside)
        hit[idx] = mask[y[idx], x[idx]] > 0
        votes += hit.astype(np.uint8)
        if i == args.reference:
            ref_uv, ref_depth, ref_inside = uv, depth, hit
    eligible = (votes >= max(3, int(np.ceil(.75 * len(frames))))) & ref_inside
    points, uids, ref_uv, ref_depth, votes = (array[eligible] for array in (points, uids, ref_uv, ref_depth, votes))
    if len(points) < args.num_landmarks:
        raise ValueError(f'Only {len(points)} multi-view MPS object points; need {args.num_landmarks}')
    np.savez_compressed(args.output / 'mps_chair_points.npz', points_world=points, uids=uids,
                        reference_uv=ref_uv, reference_depth=ref_depth, votes=votes,
                        T_world_camera_reference=camera['T_world_camera'][args.reference],
                        reference_frame=args.reference)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(cloud.cluster_dbscan(eps=.035, min_points=20, print_progress=False))
    clusters = []
    overall_depth = np.median(ref_depth)
    depth_mad = np.median(np.abs(ref_depth - overall_depth))
    for label in np.unique(labels):
        if label < 0:
            continue
        members = labels == label
        if members.sum() >= 50 and abs(np.median(ref_depth[members])-overall_depth) < 3 * max(depth_mad, .03):
            clusters.append(members)
    if not clusters:
        raise ValueError('Could not identify a coherent MPS object surface')
    keep = np.logical_or.reduce(clusters)
    clean_points, clean_uv, clean_uids = points[keep], ref_uv[keep], uids[keep]
    image = cv2.imread(str(args.sequence / f'all_frames/{args.reference:06d}.png'))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
    x = np.clip(np.rint(clean_uv[:,0]).astype(int), 0, width-1)
    y = np.clip(np.rint(clean_uv[:,1]).astype(int), 0, height-1)
    strength = np.sqrt(np.maximum(contrast[y,x], 0))
    strength /= max(np.quantile(strength, .95), 1e-9)
    strength = np.clip(strength, 0, 1)
    selected = []
    separation = np.full(len(clean_uv), np.inf)
    for _ in range(args.num_landmarks):
        score = (.3 + .7 * strength) if not selected else separation * (.3 + .7 * strength)
        if selected:
            score[separation < 4] = -np.inf
        index = int(np.argmax(score))
        if not np.isfinite(score[index]):
            raise ValueError(f'Could select only {len(selected)} spatially separated landmarks')
        selected.append(index)
        separation = np.minimum(separation, np.linalg.norm(clean_uv - clean_uv[index], axis=1))
    selected = np.asarray(selected)
    T = camera['T_world_camera'][args.reference]
    anchor_points = (clean_points[selected] - T[:3,3]) @ T[:3,:3]
    np.savez_compressed(args.output / 'landmarks.npz', points_world=clean_points[selected],
                        points_anchor=anchor_points, reference_uv=clean_uv[selected],
                        uids=clean_uids[selected], reference_frame=args.reference,
                        T_world_anchor=T, clean_points_world=clean_points,
                        clean_reference_uv=clean_uv)
    for x,y in clean_uv[::max(1,len(clean_uv)//1000)]:
        cv2.circle(image,(round(x),round(y)),1,(0,255,255),-1)
    for x,y in clean_uv[selected]:
        cv2.circle(image,(round(x),round(y)),3,(0,0,255),-1)
    cv2.imwrite(str(args.output / 'landmarks_reference.jpg'), image)
    report={'static_frames':frames,'reference_frame':args.reference,'high_quality_mps_points':int(quality.sum()),
            'multi_view_mask_points':len(points),'clean_object_surface_points':len(clean_points),
            'landmarks':len(selected),'depth_median_m':float(np.median(ref_depth[keep])),
            'selection':'multi-view mask votes, Euclidean surface clusters, spaced textured features'}
    (args.output/'landmarks_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
