#!/usr/bin/env python3
"""Camera-aware metric rigid tracking with optical flow and silhouette recovery.

Bootstrap from the metric MPS mesh/landmarks. Transform the previous pose into
each new camera before optimization. Replenish visible mesh landmarks instead
of relying forever on points visible in one reference view. Tag silhouette-only
recovery separately; do not conceal occlusion or missing camera coverage.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from camera_motion_geometry import estimate_rigid_pose, project_points, propagate_pose
from fit_mesh_to_mps import bilinear


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--landmarks',type=Path,required=True)
    parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    root=args.sequence;anchor=np.load(args.anchor);camera=np.load(root/'aria_camera_trajectory.npz');landmarks=np.load(args.landmarks);tracks=np.load(args.tracks)
    if not np.array_equal(tracks['timestamps_ns'],camera['timestamps_ns']):raise ValueError('Timestamp mismatch')
    K=anchor['K'];f=K[0,0];cx,cy=K[:2,2];n=len(camera['timestamps_ns']);reference=int(landmarks['reference_frame'])
    T_anchor_object=anchor['T_anchor_object'];inverse_anchor=np.linalg.inv(T_anchor_object)
    initial_points=landmarks['points_anchor']@inverse_anchor[:3,:3].T+inverse_anchor[:3,3]
    initial_points_valid=landmarks['triangulated_valid'] if 'triangulated_valid' in landmarks else np.ones(len(initial_points),bool)
    mesh=trimesh.load(str(anchor['fixed_mesh_path']),force='mesh',process=False);verts=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
    sample_vertices=np.unique(verts,axis=0);sample_vertices=sample_vertices[np.linspace(0,len(sample_vertices)-1,min(1400,len(sample_vertices))).astype(int)]
    scene=o3d.t.geometry.RaycastingScene(nthreads=2);scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(verts.astype(np.float32)),o3d.core.Tensor(faces.astype(np.int32))))
    images=[cv2.imread(str(root/f'all_frames/{i:06d}.png')) for i in range(n)]
    grays=[cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
    masks=[cv2.imread(str(root/f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'),0)>0 for i in range(n)]
    height,width=grays[0].shape
    values=[cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,2] for image in images]
    reference_brightness=float(np.median(values[reference][masks[reference]]))
    def replenish(i,pose,points,uv):
        mask=cv2.erode(masks[i].astype(np.uint8),np.ones((3,3),np.uint8))
        for x,y in np.rint(uv).astype(int):cv2.circle(mask,(x,y),6,0,-1)
        corners=cv2.goodFeaturesToTrack(grays[i],maxCorners=max(0,160-len(points)),qualityLevel=.01,minDistance=6,mask=mask) if len(points)<160 else None
        if corners is None:return points,uv
        new_uv=corners.reshape(-1,2)
        directions_camera=np.column_stack([(new_uv[:,0]-cx)/f,(new_uv[:,1]-cy)/f,np.ones(len(new_uv))])
        origins=np.broadcast_to(-pose[:3,:3].T@pose[:3,3],(len(new_uv),3))
        directions=directions_camera@pose[:3,:3]
        rays=np.column_stack([origins,directions]).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=2)['t_hit'].numpy()
        valid=np.isfinite(hit)&(hit>.1)&(hit<5.)
        new_points=origins[valid]+directions[valid]*hit[valid,None]
        return np.concatenate([points,new_points]),np.concatenate([uv,new_uv[valid]])

    def silhouette(pose,mask):
        uv,depth=project_points(verts,pose,K)
        rendered=np.zeros(mask.shape,np.uint8)
        for face in faces:
            if (depth[face]>.05).all():cv2.fillConvexPoly(rendered,np.rint(uv[face]).astype(np.int32),1)
        binary=rendered>0
        return float((binary&mask).sum()/max((binary|mask).sum(),1)),binary

    def recover(prior,mask):
        if mask.sum()<300:return None,0.
        outside=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=np.concatenate(contours).reshape(-1,2);contour=contour[np.linspace(0,len(contour)-1,min(180,len(contour))).astype(int)]
        def unpack(p):
            pose=prior.copy();pose[:3,:3]=prior[:3,:3]@Rotation.from_rotvec(p[:3]).as_matrix();pose[:3,3]=prior[:3,3]+p[3:];return pose
        def residual(p):
            uv,depth=project_points(sample_vertices,unpack(p),K)
            return np.r_[bilinear(outside,uv)/(4*np.sqrt(len(uv))),cKDTree(uv).query(contour)[0]/(4*np.sqrt(len(contour))),p[:3]/.25*.25,p[3:]/.06*.25,np.minimum(depth-.05,0)*100]
        fit=least_squares(residual,np.zeros(6),loss='soft_l1',f_scale=1.,diff_step=1e-4,max_nfev=45,ftol=1e-5,xtol=1e-5)
        pose=unpack(fit.x);score,_=silhouette(pose,mask)
        return pose,score

    T_camera=np.full((n,4,4),np.nan);T_world=np.full_like(T_camera,np.nan);valid=np.zeros(n,bool);world_valid=np.zeros(n,bool);confidence=np.zeros(n);rows={}
    T_camera[reference]=T_anchor_object;valid[reference]=True;T_world[reference]=camera['T_world_camera'][reference]@T_anchor_object;world_valid[reference]=True
    rows[reference]={'frame':reference,'method':'metric_anchor','inliers':int(initial_points_valid.sum()),'reprojection_rms_px':0.,'silhouette_iou':silhouette(T_anchor_object,masks[reference])[0],'valid':True,'world_valid':True,'rejection_reason':''};confidence[reference]=1.
    for direction,end in [(1,n),(-1,-1)]:
        previous=reference;pose=T_anchor_object.copy();points=initial_points[initial_points_valid].copy();uv=tracks['tracks_uv'][initial_points_valid,reference].copy();points,uv=replenish(reference,pose,points,uv)
        for i in range(reference+direction,end,direction):
            mask_pixels=int(masks[i].sum())
            brightness=float(np.median(values[i][masks[i]])) if mask_pixels else float('nan')
            # SAM can switch from the white chair to the nearby grey chair when
            # the target leaves view. A convincing PnP fit to that distractor
            # is still the wrong object, so reject the mask before tracking.
            reason='insufficient_visible_area' if mask_pixels<300 else 'appearance_mismatch' if brightness<.45*reference_brightness else ''
            if reason:
                rows[i]={'frame':i,'method':'invalid','inliers':0,'reprojection_rms_px':float('nan'),'silhouette_iou':float('nan'),'valid':False,'world_valid':False,'rejection_reason':reason}
                print(f'frame {i}: invalid ({reason}); mask pixels={mask_pixels}, median V={brightness:.1f}',flush=True)
                continue
            prior=propagate_pose(camera['T_world_camera'][previous],camera['T_world_camera'][i],pose) if camera['valid'][previous] and camera['valid'][i] else pose.copy()
            pnp_pose=None;info={'status':'insufficient_correspondences','inliers':0}
            tracked=np.empty((0,2));local=np.empty((0,3))
            if len(uv)>=8:
                tracked,status,error=cv2.calcOpticalFlowPyrLK(grays[previous],grays[i],uv.astype(np.float32).reshape(-1,1,2),None,winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                back,back_status,_=cv2.calcOpticalFlowPyrLK(grays[i],grays[previous],tracked,None,winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                tracked=tracked.reshape(-1,2);fb=np.linalg.norm(back.reshape(-1,2)-uv,axis=1)
                use=status.reshape(-1).astype(bool)&back_status.reshape(-1).astype(bool)&(fb<1.5)&(tracked[:,0]>=0)&(tracked[:,0]<width)&(tracked[:,1]>=0)&(tracked[:,1]<height)
                if masks[i].any():
                    mask=cv2.dilate(masks[i].astype(np.uint8),np.ones((9,9),np.uint8));indices=np.flatnonzero(use);xy=np.rint(tracked[indices]).astype(int);xy[:,0]=np.clip(xy[:,0],0,width-1);xy[:,1]=np.clip(xy[:,1],0,height-1);use[indices]&=mask[xy[:,1],xy[:,0]]>0
                local,tracked=points[use],tracked[use]
                pnp_pose,info=estimate_rigid_pose(local,tracked,K,prior,min_inliers=8)
            candidate=pnp_pose;method='flow_pnp'
            if candidate is not None:
                fit_score,_=silhouette(candidate,masks[i]) if masks[i].any() else (float('nan'),None)
                if np.isfinite(fit_score) and fit_score<.4:candidate=None
            if candidate is None:
                candidate,fit_score=recover(prior,masks[i]);method='silhouette_recovery'
                if candidate is not None and fit_score<.4:candidate=None
            reason=''
            if candidate is not None and method=='silhouette_recovery' and abs(i-previous)>2:
                # A silhouette alone cannot reidentify a small, partly
                # occluded chair after a long visibility gap. Require an
                # independent landmark match before starting a new segment.
                candidate=None;reason='recovery_after_long_gap_unverified'
            if candidate is not None and camera['valid'][i] and camera['valid'][previous]:
                current_world=camera['T_world_camera'][i]@candidate
                previous_world=camera['T_world_camera'][previous]@pose
                elapsed=abs(i-previous)
                displacement=float(np.linalg.norm(current_world[:3,3]-previous_world[:3,3]))
                angle=float(np.degrees(Rotation.from_matrix(previous_world[:3,:3].T@current_world[:3,:3]).magnitude()))
                # This is a generous plausibility bound, not a stationary-object
                # prior: genuine chair movement in this clip peaks below 0.09 m
                # and 13 degrees per 0.1 s in the well-observed interval.
                if displacement>.15*elapsed or angle>20.*elapsed:
                    candidate=None;reason='implausible_world_jump'
            row={'frame':i,'method':method if candidate is not None else 'invalid','inliers':info.get('inliers',0),'reprojection_rms_px':info.get('reprojection_rms_px',float('nan')),'silhouette_iou':fit_score if candidate is not None else float('nan'),'valid':candidate is not None,'world_valid':bool(candidate is not None and camera['valid'][i]),'rejection_reason':reason}
            rows[i]=row
            if candidate is None:
                # Retain the last measured world pose only as an optimizer prior.
                # This frame remains NaN; never export a held pose as a measurement.
                continue
            pose=candidate;T_camera[i]=pose;valid[i]=True
            if camera['valid'][i]:T_world[i]=camera['T_world_camera'][i]@pose;world_valid[i]=True
            confidence[i]=min(1.,info.get('inlier_ratio',0.)*np.exp(-info.get('reprojection_rms_px',3.)**2/18.)) if method=='flow_pnp' else float(fit_score*.35)
            if len(local):
                projected,_=project_points(local,pose,K);good=np.linalg.norm(projected-tracked,axis=1)<3.;points,uv=local[good],tracked[good]
            else:points,uv=np.empty((0,3)),np.empty((0,2))
            if method!='flow_pnp' or len(points)<60 or i%8==0:points,uv=replenish(i,pose,points,uv)
            previous=i
            print(f'frame {i}: {method}, points={len(points)}, inliers={row["inliers"]}, IoU={fit_score:.3f}',flush=True)
    quats=np.full((n,4),np.nan);quats[valid]=Rotation.from_matrix(T_camera[valid,:3,:3]).as_quat()[:,[3,0,1,2]]
    np.savez_compressed(out/'room4_whitechair_trajectory_mps.npz',T_camera_object=T_camera,T_world_object=T_world,transforms=T_camera,translations=T_camera[:,:3,3],rotations=T_camera[:,:3,:3],quaternions_wxyz=quats,valid=valid,world_valid=world_valid,confidence=confidence,timestamps_ns=camera['timestamps_ns'],timestamps=camera['timestamps_ns']*1e-9,relative_timestamps=(camera['timestamps_ns']-camera['timestamps_ns'][0])*1e-9,frame_indices=np.arange(n),T_world_camera=camera['T_world_camera'],camera_valid=camera['valid'],K=K,fixed_mesh_path=str(anchor['fixed_mesh_path']),translation_units='meters',metric_scale_verified=True,metric_scale_source=str(anchor['scale_source']),coordinate_frame='OpenCV camera; MPS world',transform_convention='column vectors; metric mesh scale baked in',reference_frame=reference,method=np.array([rows[i]['method'] for i in range(n)]))
    with (out/'frame_metrics.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[reference]));writer.writeheader();writer.writerows(rows[i] for i in range(n))
    static=[i for i in range(20,56) if world_valid[i]];ref=T_world[reference]
    def stat(a):
        a=np.asarray(a);a=a[np.isfinite(a)];return {'count':len(a),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max())} if len(a) else {}
    steps=[float(np.degrees(Rotation.from_matrix(T_world[i-1,:3,:3].T@T_world[i,:3,:3]).magnitude())) for i in range(21,56) if world_valid[i-1] and world_valid[i]]
    report={'frames':n,'camera_valid':int(valid.sum()),'world_valid':int(world_valid.sum()),'invalid_camera_frames':np.flatnonzero(~valid).tolist(),'invalid_world_frames':np.flatnonzero(~world_valid).tolist(),'method_counts':{method:sum(r['method']==method for r in rows.values()) for method in set(r['method'] for r in rows.values())},'rejection_counts':{reason:sum(r['rejection_reason']==reason for r in rows.values()) for reason in set(r['rejection_reason'] for r in rows.values()) if reason},'static_translation_from_reference_m':stat([np.linalg.norm(T_world[i,:3,3]-ref[:3,3]) for i in static]),'static_rotation_from_reference_deg':stat([np.degrees(Rotation.from_matrix(ref[:3,:3].T@T_world[i,:3,:3]).magnitude()) for i in static]),'static_world_rotation_steps_deg':stat(steps),'silhouette_iou':stat([r['silhouette_iou'] for r in rows.values()]),'notes':['Camera-aware priors use actual MPS SE(3) motion in meters.','Silhouette recovery is tagged and assigned lower confidence than PnP.','No stationarity constraint is applied after metric anchor calibration.','Grey-chair segmentation switches and implausible world jumps are rejected.','Missing measurements remain invalid; world poses require valid MPS coverage.']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
