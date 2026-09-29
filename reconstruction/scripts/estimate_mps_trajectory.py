#!/usr/bin/env python3
"""Recover metric camera/world chair poses from MPS-backed TAPIR landmarks.

Camera movement is compensated in the prior and world-pose composition. A
failed measurement stays invalid; it is not replaced by a frozen object pose.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_motion_geometry import estimate_rigid_pose


def stat(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return {'count':len(a),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max())} if len(a) else {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--landmarks',type=Path,required=True)
    parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--anchor',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((args.sequence/'aria_metadata.json').read_text());f=metadata['focal_px'];cx,cy=metadata['principal_point_px'];width,height=metadata['pinhole_size']
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
    landmarks=np.load(args.landmarks);tracks=np.load(args.tracks);camera=np.load(args.sequence/'aria_camera_trajectory.npz')
    if not np.array_equal(tracks['timestamps_ns'],camera['timestamps_ns']):raise ValueError('Camera and track timestamps differ')
    pixels=tracks['tracks_uv'];confidence=tracks['confidence'];points=landmarks['points_anchor'];n=len(camera['timestamps_ns'])
    world_anchor=landmarks['T_world_anchor']
    T_camera_anchor=np.full((n,4,4),np.nan);T_world_estimated_anchor=np.full_like(T_camera_anchor,np.nan)
    valid=np.zeros(n,bool);world_valid=np.zeros(n,bool);scores=np.zeros(n);rows=[]
    previous_world=world_anchor.copy()
    # Forward then backward from the actual reference, with independent chains.
    reference=int(landmarks['reference_frame'])
    order=list(range(reference,n))+list(range(reference-1,-1,-1))
    for i in order:
        if i==reference-1:previous_world=world_anchor.copy()
        uv=pixels[:,i];conf=confidence[:,i]
        use=(conf>.5)&np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        if 'triangulated_valid' in landmarks:
            use &= landmarks['triangulated_valid']
        mask=cv2.imread(str(args.sequence/f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'),0)
        if mask is not None and mask.any():
            mask=cv2.dilate((mask>0).astype(np.uint8),np.ones((11,11),np.uint8))
            indices=np.flatnonzero(use);xy=np.rint(uv[indices]).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,width-1);xy[:,1]=np.clip(xy[:,1],0,height-1)
            use[indices]&=mask[xy[:,1],xy[:,0]]>0
        prior=np.linalg.inv(camera['T_world_camera'][i])@previous_world if camera['valid'][i] else None
        pose,info=estimate_rigid_pose(points[use],uv[use],K,prior=prior)
        row={'frame':i,'time_s':float((camera['timestamps_ns'][i]-camera['timestamps_ns'][0])*1e-9),'visible_landmarks':int(use.sum()),'camera_mps_valid':bool(camera['valid'][i]),'status':info['status'],'inliers':info.get('inliers',0),'inlier_ratio':info.get('inlier_ratio',0.),'reprojection_rms_px':info.get('reprojection_rms_px',float('nan'))}
        if pose is not None:
            valid[i]=True;T_camera_anchor[i]=pose
            scores[i]=float(np.mean(conf[use])*info['inlier_ratio']*np.exp(-info['reprojection_rms_px']**2/18.))
            if camera['valid'][i]:
                world_valid[i]=True;T_world_estimated_anchor[i]=camera['T_world_camera'][i]@pose
                previous_world=T_world_estimated_anchor[i]
        rows.append(row)
    rows.sort(key=lambda x:x['frame'])
    np.savez_compressed(out/'rigid_anchor_trajectory.npz',T_camera_anchor=T_camera_anchor,T_world_anchor=T_world_estimated_anchor,valid=valid,world_valid=world_valid,confidence=scores,timestamps_ns=camera['timestamps_ns'],K=K,translation_units='meters',reference_frame=reference)
    T_anchor_object=np.eye(4);fixed_mesh_path='';scale_source='MPS metric landmarks; object frame is the reference RGB camera'
    if args.anchor:
        anchor=np.load(args.anchor);T_anchor_object=anchor['T_anchor_object'];fixed_mesh_path=str(anchor['fixed_mesh_path']);scale_source=str(anchor['scale_source'])
    T_camera_object=T_camera_anchor@T_anchor_object
    T_world_object=T_world_estimated_anchor@T_anchor_object
    quaternions=np.full((n,4),np.nan)
    quaternions[valid]=Rotation.from_matrix(T_camera_object[valid,:3,:3]).as_quat()[:,[3,0,1,2]]
    np.savez_compressed(out/'room4_whitechair_trajectory_mps.npz',T_camera_object=T_camera_object,T_world_object=T_world_object,transforms=T_camera_object,translations=T_camera_object[:,:3,3],rotations=T_camera_object[:,:3,:3],quaternions_wxyz=quaternions,valid=valid,world_valid=world_valid,confidence=scores,timestamps_ns=camera['timestamps_ns'],timestamps=camera['timestamps_ns']*1e-9,relative_timestamps=(camera['timestamps_ns']-camera['timestamps_ns'][0])*1e-9,frame_indices=np.arange(n),T_world_camera=camera['T_world_camera'],camera_valid=camera['valid'],K=K,fixed_mesh_path=fixed_mesh_path,translation_units='meters',metric_scale_verified=True,metric_scale_source=scale_source,coordinate_frame='OpenCV camera; MPS world',transform_convention='column vectors; mesh scale baked in',reference_frame=reference)
    ref=T_world_object[reference]
    static_indices=np.array([i for i in range(20,56) if world_valid[i]])
    static_translation=np.linalg.norm(T_world_object[static_indices,:3,3]-ref[:3,3],axis=1)
    static_angles=Rotation.from_matrix(np.swapaxes(ref[:3,:3],-1,-2)@T_world_object[static_indices,:3,:3]).magnitude()*180/np.pi
    steps=[]
    for i in range(1,n):
        if world_valid[i-1] and world_valid[i]:
            angle=Rotation.from_matrix(T_world_object[i-1,:3,:3].T@T_world_object[i,:3,:3]).magnitude()*180/np.pi
            steps.append({'from_frame':i-1,'to_frame':i,'translation_step_m':float(np.linalg.norm(T_world_object[i,:3,3]-T_world_object[i-1,:3,3])),'rotation_step_deg':float(angle)})
    static_steps=[s for s in steps if 20<=s['from_frame'] and s['to_frame']<=55]
    report={'frames':n,'camera_pose_valid':int(valid.sum()),'world_pose_valid':int(world_valid.sum()),'invalid_camera_pose_frames':np.flatnonzero(~valid).tolist(),'invalid_world_pose_frames':np.flatnonzero(~world_valid).tolist(),'focal_px':f,'metric_scale_source':scale_source,'reference_frame':reference,'reprojection_rms_px':stat([r['reprojection_rms_px'] for r in rows]),'static_frames_20_55_translation_from_reference_m':stat(static_translation),'static_frames_20_55_rotation_from_reference_deg':stat(static_angles),'static_frames_20_55_rotation_steps_deg':stat([s['rotation_step_deg'] for s in static_steps]),'static_frames_20_55_translation_steps_m':stat([s['translation_step_m'] for s in static_steps]),'world_rotation_steps_deg':stat([s['rotation_step_deg'] for s in steps]),'world_translation_steps_m':stat([s['translation_step_m'] for s in steps]),'largest_world_rotation_jumps':sorted(steps,key=lambda s:s['rotation_step_deg'],reverse=True)[:12],'motion_from_reference_translation_m':stat([np.linalg.norm(T_world_object[i,:3,3]-ref[:3,3]) for i in range(n) if world_valid[i]]),'notes':['No stationary-object constraint is applied to PnP; actual object motion is retained.','Invalid estimates remain NaN; they are never replaced by a constant world pose.','Static evaluation uses the visually pre-manipulation interval 20–55.','MPS coverage is missing for frames 0–11 and 306; camera-relative PnP may still be valid there.']}
    (out/'trajectory_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'trajectory_frame_metrics.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
