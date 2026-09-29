#!/usr/bin/env python3
"""Render RGB/mesh overlay beside a fixed-scale MPS room trajectory map."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import trimesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=np.load(args.trajectory);rows=list(csv.DictReader((args.trajectory.parent/'frame_metrics.csv').open()))
    mesh=trimesh.load(str(data['fixed_mesh_path']),force='mesh',process=False)
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);K=data['K'];n=len(data['valid'])
    camera=data['T_world_camera'];obj=data['T_world_object'];camera_valid=data['camera_valid'];valid=data['world_valid']
    positions=np.concatenate([camera[camera_valid,:3,3],obj[valid,:3,3]])
    lo=positions[:,:2].min(axis=0)-.35;hi=positions[:,:2].max(axis=0)+.35
    center=(lo+hi)/2;span=max(hi-lo);lo=center-span/2;hi=center+span/2
    size=576;margin=45;scale=(size-2*margin)/span
    def xy(point):
        return tuple(np.rint([(point[0]-lo[0])*scale,size-(point[1]-lo[1])*scale]).astype(int))
    def label(canvas,text,y,color=(220,220,220),size=.55):
        cv2.putText(canvas,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,2,cv2.LINE_AA)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),10,(1344,576))
    if not writer.isOpened():raise RuntimeError('OpenCV could not open MP4 writer')
    for i in range(n):
        img=cv2.imread(str(args.sequence/f'all_frames/{i:06d}.png'))
        mask=cv2.imread(str(args.sequence/f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'),0)
        if mask is not None and mask.any():
            contours,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(30,230,20),2)
        if data['valid'][i]:
            pc=vertices@data['T_camera_object'][i,:3,:3].T+data['T_camera_object'][i,:3,3]
            uv=pc[:,:2]/np.maximum(pc[:,2,None],.05)*K[0,0]+K[:2,2]
            raster=np.zeros(img.shape[:2],np.uint8)
            for face in faces:
                if (pc[face,2]>.05).all():cv2.fillConvexPoly(raster,np.rint(uv[face]).astype(np.int32),1)
            contours,_=cv2.findContours(raster,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(20,20,250),2)
        cv2.rectangle(img,(0,0),(767,60),(10,10,10),-1)
        label(img,f'Frame {i:03d}  {data["relative_timestamps"][i]:05.2f}s  |  RGB Aria',29)
        label(img,'SAM verde  /  mesh metrica rossa',53,(180,220,180),.48)
        map_image=np.full((size,size,3),(25,30,38),np.uint8)
        for value in np.arange(np.ceil(lo[0]*2)/2,hi[0],.5):
            x=xy((value,lo[1]))[0];cv2.line(map_image,(x,0),(x,size),(48,52,60),1)
        for value in np.arange(np.ceil(lo[1]*2)/2,hi[1],.5):
            y=xy((lo[0],value))[1];cv2.line(map_image,(0,y),(size,y),(48,52,60),1)
        for name,poses,available,color in [('camera',camera,camera_valid,(210,100,160)),('chair',obj,valid,(245,150,40))]:
            for j in range(1,i+1):
                if available[j] and available[j-1]:cv2.line(map_image,xy(poses[j-1,:3,3]),xy(poses[j,:3,3]),color,3,cv2.LINE_AA)
        if camera_valid[i]:
            cp=xy(camera[i,:3,3]);cv2.circle(map_image,cp,8,(210,100,160),-1)
            forward=camera[i,:3,2][:2];norm=np.linalg.norm(forward)
            if norm>.08:cv2.arrowedLine(map_image,cp,xy(camera[i,:3,3]+np.r_[forward/norm*.2,0]),(210,100,160),2,tipLength=.3)
        if valid[i]:
            op=xy(obj[i,:3,3]);cv2.circle(map_image,op,9,(245,150,40),-1)
            axis=obj[i,:2,0].copy();axis/=max(np.linalg.norm(axis),1e-6)
            cv2.arrowedLine(map_image,op,xy(obj[i,:3,3]+np.r_[axis*.17,0]),(245,150,40),2,tipLength=.3)
        cv2.circle(map_image,xy(obj[28,:3,3]),5,(90,200,80),-1)
        cv2.rectangle(map_image,(0,0),(size,111),(15,18,23),-1)
        label(map_image,'MONDO MPS - vista dall alto',28,(235,235,235),.62)
        label(map_image,'Camera Aria viola  |  sedia blu  |  origine verde',55,(220,220,220),.42)
        label(map_image,f'Sedia: {"valida" if valid[i] else "non misurata"}  Metodo: {rows[i]["method"]}',80,(220,220,220),.44)
        if valid[i]:
            displacement=np.linalg.norm(obj[i,:3,3]-obj[28,:3,3]);label(map_image,f'Spostamento rispetto al frame 28: {displacement:.2f} m',103,(220,220,220),.45)
        else:label(map_image,rows[i]['rejection_reason'] or 'Camera MPS assente / misura non valida',103,(105,180,240),.43)
        bar=np.concatenate((img,map_image),axis=1)
        writer.write(bar)
        if i%50==0:print(f'Rendered {i}/{n}',flush=True)
    writer.release()
    print(args.output)


if __name__=='__main__':main()
