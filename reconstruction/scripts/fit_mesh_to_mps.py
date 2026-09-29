#!/usr/bin/env python3
"""Fit a fixed SAM3D mesh to metric MPS chair surfaces and calibrated masks.

Only the pre-manipulation anchor geometry is fitted here. Object motion is
estimated independently by PnP and is never forced to remain stationary.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from camera_motion_geometry import project_points


def bilinear(image, uv):
    height, width = image.shape
    x = np.clip(uv[:, 0], 0, width - 1.001)
    y = np.clip(uv[:, 1], 0, height - 1.001)
    ix, iy = x.astype(int), y.astype(int)
    dx, dy = x - ix, y - iy
    value = (1-dx)*(1-dy)*image[iy, ix] + dx*(1-dy)*image[iy, ix+1] + (1-dx)*dy*image[iy+1, ix] + dx*dy*image[iy+1, ix+1]
    return value + np.linalg.norm(uv - np.column_stack([x, y]), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--landmarks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.sequence, args.output
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / 'aria_metadata.json').read_text())
    focal = metadata['focal_px'];cx, cy = metadata['principal_point_px']
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]])
    camera = np.load(root / 'aria_camera_trajectory.npz')
    landmarks = np.load(args.landmarks)
    points = landmarks['clean_points_world']
    points = points[np.linspace(0, len(points)-1, min(1200, len(points))).astype(int)]
    mesh_path = root / 'video_segmentation/masks/frame_000028_masks/whitechair/whitechair.obj'
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = np.asarray(mesh.vertices)
    unique_vertices = np.unique(vertices, axis=0)
    sampled_vertices = unique_vertices[np.linspace(0, len(unique_vertices)-1, min(2000, len(unique_vertices))).astype(int)]
    canonical = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(mesh.faces).astype(np.int32)))
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(canonical)
    init = json.loads((root / 'video_segmentation/masks/frame_000028_masks/layout.json').read_text())['objects'][0]['local_to_scene']
    A = np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    D = np.diag([-1.,-1.,1.]);q = np.asarray(init['quat_wxyz'])
    R_camera = D @ Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T @ A
    translation_camera = D @ np.asarray(init['translation'])
    depth_metric = landmarks['points_anchor'][:,2]
    units = float(np.median(depth_metric) / translation_camera[2])
    scale = float(init['scale'][0]) * units
    T = camera['T_world_camera'][28]
    R_world = T[:3,:3] @ R_camera
    translation_world = T[:3,:3] @ (translation_camera * units) + T[:3,3]
    initial = np.r_[Rotation.from_matrix(R_world).as_rotvec(), translation_world, np.log(scale)]
    observations = []
    for i in [20, 28, 36]:
        mask = cv2.imread(str(root / f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'),0)>0
        contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour = np.concatenate(contours).reshape(-1,2)
        contour = contour[np.linspace(0,len(contour)-1,min(150,len(contour))).astype(int)]
        outside = cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        observations.append((i, np.linalg.inv(camera['T_world_camera'][i]), outside, contour))

    calls = 0
    start = time.monotonic()
    def residual(parameters):
        nonlocal calls
        calls += 1
        R = Rotation.from_rotvec(parameters[:3]).as_matrix()
        position = parameters[3:6];scale = np.exp(parameters[6])
        local = (points - position) @ R / scale
        distances = scene.compute_distance(o3d.core.Tensor(local.astype(np.float32)), nthreads=2).numpy()*scale
        result = [distances/.02]
        verts_world = sampled_vertices*scale @ R.T + position
        for i, T_camera_world, outside, contour in observations:
            uv, depth = project_points(verts_world,T_camera_world,K)
            result.append(bilinear(outside,uv)/5.)
            result.append(cKDTree(uv).query(contour)[0]/5.)
        # Preserve the well-supported orientation of the reference mesh while
        # allowing correction; metric surfaces and masks supply the main data.
        result.append((parameters[:3]-initial[:3])/.5)
        if calls%100==0:print(f'fit evaluations={calls}, elapsed={time.monotonic()-start:.1f}s, surface median={np.median(distances):.4f}m',flush=True)
        return np.concatenate(result)

    print(f'Fitting metric mesh: {len(points)} surface points; initial scale {scale:.4f}m',flush=True)
    optimized = least_squares(residual,initial,loss='soft_l1',f_scale=1.,diff_step=1e-4,max_nfev=160,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    params = optimized.x
    R = Rotation.from_rotvec(params[:3]).as_matrix();position=params[3:6];scale=np.exp(params[6])
    T_world_object=np.eye(4);T_world_object[:3,:3]=R;T_world_object[:3,3]=position
    T_anchor_object=np.linalg.inv(camera['T_world_camera'][28])@T_world_object
    distances=scene.compute_distance(o3d.core.Tensor(((points-position)@R/scale).astype(np.float32)),nthreads=2).numpy()*scale
    mesh.vertices=vertices*scale
    fixed_mesh=out/'whitechair_metric.obj';mesh.export(fixed_mesh)
    np.savez_compressed(out/'mesh_anchor.npz',T_world_object_reference=T_world_object,T_anchor_object=T_anchor_object,fixed_scale_m=scale,K=K,fixed_mesh_path=str(fixed_mesh.resolve()),translation_units='meters',scale_source='Aria MPS semidense chair surfaces and calibrated multi-view silhouettes')
    print('FIT',optimized.message,'scale',scale,'distance median',np.median(distances),'p90',np.quantile(distances,.9),flush=True)
    results=[]
    for i in [20,24,28,32,36,40,44,48,52,55]:
        target=cv2.imread(str(root/f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'),0)>0
        pc=vertices*scale@R.T+position
        uv,z=project_points(pc,np.linalg.inv(camera['T_world_camera'][i]),K)
        triangles=np.rint(uv).astype(np.int32)[mesh.faces]
        rendered=np.zeros(target.shape,np.uint8)
        for triangle in triangles:cv2.fillConvexPoly(rendered,triangle,1)
        pred=rendered>0
        score=float((pred&target).sum()/max((pred|target).sum(),1))
        results.append({'frame':i,'silhouette_iou':score,'used_in_fit':i in [20,28,36]})
        image=cv2.imread(str(root/f'all_frames/{i:06d}.png'))
        for mask,color in [(target,(0,255,0)),(pred,(0,0,255))]:
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,contours,-1,color,2)
        cv2.putText(image,f'frame {i} | metric static anchor | IoU {score:.3f}',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        cv2.imwrite(str(out/f'anchor_fit_{i:06d}.jpg'),image)
    report={'runtime_s':time.monotonic()-start,'optimizer_success':bool(optimized.success),'optimizer_message':optimized.message,'canonical_mesh_scale_m':float(scale),'surface_distance_median_m':float(np.median(distances)),'surface_distance_p90_m':float(np.quantile(distances,.9)),'silhouette_checks':results,'held_out_iou_median':float(np.median([r['silhouette_iou'] for r in results if not r['used_in_fit']])),'notes':['Metric geometry comes from MPS, independently of MoGe depth scale.','Only anchor geometry uses pre-manipulation views; motion is separately estimated from tracked landmarks.','CPU triangle rasterization is an approximate silhouette check; masks are model predictions rather than manual annotations.']}
    (out/'mesh_fit_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
