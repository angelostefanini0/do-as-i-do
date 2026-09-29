#!/usr/bin/env python3
"""Refine chair landmarks by robust calibrated multi-view triangulation.

Use only a verified pre-manipulation interval for landmark calibration. This
does not impose stationarity on subsequent object-pose estimation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',required=True,type=Path)
    parser.add_argument('--landmarks',required=True,type=Path)
    parser.add_argument('--tracks',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--static-frames',default='20:56')
    args=parser.parse_args()
    source=np.load(args.landmarks);tracks=np.load(args.tracks);camera=np.load(args.sequence/'aria_camera_trajectory.npz');meta=json.loads((args.sequence/'aria_metadata.json').read_text());f=meta['focal_px'];cx,cy=meta['principal_point_px']
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
    start,end=map(int,args.static_frames.split(':'));frames=np.array([i for i in range(start,end) if camera['valid'][i]])
    projections=K[None]@np.linalg.inv(camera['T_world_camera'][frames])[:,:3,:]
    points=source['points_world'].copy();valid=[];rows=[]
    for j in range(len(points)):
        confidence=tracks['confidence'][j,frames];use=confidence>.6
        uv=tracks['tracks_uv'][j,frames][use];P=projections[use];weights=np.sqrt(confidence[use])
        if len(uv)<12:rows.append({'landmark':j,'status':'insufficient_static_views'});continue
        A=np.concatenate([uv[:,0,None]*P[:,2]-P[:,0],uv[:,1,None]*P[:,2]-P[:,1]])
        _,_,vt=np.linalg.svd(A,full_matrices=False);initial=vt[-1,:3]/vt[-1,3]
        def residual(point):
            image=P@np.r_[point,1.];projected=image[:,:2]/image[:,2,None]
            return ((projected-uv)*weights[:,None]).reshape(-1)
        fit=least_squares(residual,initial,loss='soft_l1',f_scale=1.,max_nfev=100)
        image=P@np.r_[fit.x,1.];error=np.linalg.norm(image[:,:2]/image[:,2,None]-uv,axis=1)
        ok=np.isfinite(fit.x).all() and (image[:,2]>0).all() and np.median(error)<1.5 and np.quantile(error,.9)<3.5
        row={'landmark':j,'status':'ok' if ok else 'rejected','views':len(uv),'median_error_px':float(np.median(error)),'p90_error_px':float(np.quantile(error,.9)),'mps_point_correction_m':float(np.linalg.norm(fit.x-points[j]))}
        rows.append(row)
        if ok:points[j]=fit.x;valid.append(j)
    valid=np.array(valid);T=source['T_world_anchor'];local=(points-T[:3,3])@T[:3,:3]
    # Keep indexing unchanged; rejected landmarks are explicitly disabled in
    # downstream tracking rather than reassigning TAPIR correspondences.
    data={k:source[k] for k in source.files};data.update(points_world=points,points_anchor=local,triangulated_valid=np.isin(np.arange(len(points)),valid),landmark_scale_source='Calibrated Aria MPS multi-view triangulation of fixed chair features')
    np.savez_compressed(args.output,**data)
    report={'static_frame_range':[start,end],'total_landmarks':len(points),'valid_triangulated_landmarks':len(valid),'median_reprojection_error_px':float(np.median([x['median_error_px'] for x in rows if x['status']=='ok'])),'median_correction_to_mps_points_m':float(np.median([x['mps_point_correction_m'] for x in rows if x['status']=='ok'])),'landmarks':rows}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='landmarks'},indent=2))


if __name__=='__main__':main()
