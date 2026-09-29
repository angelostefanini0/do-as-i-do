#!/usr/bin/env python3
"""Inspect the metric object and moving Aria camera in one MPS world frame."""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8081)
    parser.add_argument('--host',default='127.0.0.1')
    args=parser.parse_args()
    data=np.load(args.trajectory)
    count=len(data['valid']);camera=data['T_world_camera'];object_poses=data['T_world_object']
    valid=data['world_valid'];camera_valid=data['camera_valid']
    rows=list(csv.DictReader(args.trajectory.parent.joinpath('frame_metrics.csv').open()))
    mesh=trimesh.load(str(data['fixed_mesh_path']),force='mesh',process=False)
    height,width=cv2.imread(str(args.sequence/'all_frames/000000.png')).shape[:2]
    focal=data['K'][0,0]
    ref=object_poses[28,:3,3]
    server=viser.ViserServer(host=args.host,port=args.port,label='Room4 white chair — MPS world')
    server.scene.set_up_direction('+z')
    server.scene.set_background_image(np.full((16,16,3),(29,35,45),dtype=np.uint8))
    server.scene.add_grid('/metric_floor_grid',width=4,height=4,cell_size=.25,section_size=1.,
                          cell_color=(80,90,105),section_color=(110,120,135),
                          plane_color=(60,70,85),plane_opacity=.12,
                          position=(float(ref[0]),float(ref[1]),-1.5))
    server.gui.add_markdown('**White chair in coordinate metriche MPS**\n\nViola: camera Aria in movimento. Blu: traiettoria della sedia. Verde: punti MPS della sedia nella posa iniziale. Le pose non misurate sono nascoste.')
    frame=server.gui.add_slider('Frame RGB originale',min=0,max=count-1,step=1,initial_value=28)
    play=server.gui.add_checkbox('Riproduci',False)
    fps=server.gui.add_slider('Frame al secondo',min=1,max=20,step=1,initial_value=10)
    show_paths=server.gui.add_checkbox('Mostra traiettorie',True)
    follow=server.gui.add_checkbox('Vista dalla camera Aria',False)
    image=server.gui.add_image(np.zeros((288,384,3),np.uint8),label='RGB, SAM verde, mesh metrica rossa',format='jpeg')
    status=server.gui.add_markdown('')
    server.scene.add_frame('/world',axes_length=.25,axes_radius=.005)
    points_path=args.trajectory.parent.parent/'landmarks.npz'
    if points_path.exists():
        pts=np.load(points_path)['clean_points_world'][::3]
        server.scene.add_point_cloud('/reference_mps_chair_surface',pts,(70,190,90),point_size=.004)
    handles=[]
    for name,positions,available,color in [('camera',camera[:,:3,3],camera_valid,(130,80,215)),('chair',object_poses[:,:3,3],valid,(30,120,240))]:
        adjacent=available[:-1]&available[1:]
        segments=np.stack((positions[:-1][adjacent],positions[1:][adjacent]),axis=1)
        handles.append(server.scene.add_line_segments(f'/{name}_trajectory',segments,np.asarray(color,np.uint8),line_width=3))
        handles.append(server.scene.add_point_cloud(f'/{name}_trajectory_samples',positions[available],color,point_size=.007))
    object_handle=server.scene.add_mesh_trimesh('/white_chair_metric',mesh)
    object_frame=server.scene.add_frame('/white_chair_axes',axes_length=.15,axes_radius=.004)
    camera_handle=server.scene.add_camera_frustum('/aria_rgb',fov=2*np.arctan(height/(2*focal)),aspect=width/height,scale=.18,color=(175,95,220))
    cv_to_gl=np.diag([1.,-1.,-1.])

    @server.on_client_connect
    def setup_camera(client):
        client.camera.up_direction=(0.,0.,1.)
        client.camera.position=tuple(ref+np.array([2.,-2.,1.7]))
        client.camera.look_at=tuple(ref)

    def update(_=None):
        i=int(frame.value)
        for handle in handles:handle.visible=show_paths.value
        img=cv2.imread(str(args.sequence/f'all_frames/{i:06d}.png'))
        mask=cv2.imread(str(args.sequence/f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'),0)
        if mask is not None and mask.any():
            contours,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(10,230,10),2)
        if bool(data['valid'][i]):
            mesh_camera=np.asarray(mesh.vertices)@data['T_camera_object'][i,:3,:3].T+data['T_camera_object'][i,:3,3]
            good=mesh_camera[:,2]>.05
            uv=mesh_camera[:,:2]/np.maximum(mesh_camera[:,2,None],.05)
            uv=uv*focal+data['K'][:2,2]
            raster=np.zeros((height,width),np.uint8)
            for face in np.asarray(mesh.faces):
                if good[face].all():cv2.fillConvexPoly(raster,np.rint(uv[face]).astype(np.int32),1)
            contours,_=cv2.findContours(raster,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(20,20,255),2)
        image.image=cv2.cvtColor(cv2.resize(img,(384,288)),cv2.COLOR_BGR2RGB)
        object_handle.visible=object_frame.visible=bool(valid[i])
        camera_handle.visible=bool(camera_valid[i])
        with server.atomic():
            if valid[i]:
                world=object_poses[i];q=Rotation.from_matrix(world[:3,:3]).as_quat()[[3,0,1,2]]
                object_handle.position=object_frame.position=world[:3,3]
                object_handle.wxyz=object_frame.wxyz=q
            if camera_valid[i]:
                world=camera[i];q=Rotation.from_matrix(world[:3,:3]@cv_to_gl).as_quat()[[3,0,1,2]]
                camera_handle.position=world[:3,3];camera_handle.wxyz=q
        if follow.value and camera_valid[i]:
            world=camera[i]
            for client in server.get_clients().values():
                client.camera.up_direction=tuple(-world[:3,1])
                client.camera.position=tuple(world[:3,3])
                client.camera.look_at=tuple(world[:3,3]+world[:3,2])
        row=rows[i]
        method=row['method'];quality=f'IoU {float(row["silhouette_iou"]):.2f}; confidenza {data["confidence"][i]:.2f}' if valid[i] else row['rejection_reason'] or 'nessuna misura valida'
        if method=='silhouette_recovery':quality+=' · ricostruzione solo da sagoma, cautela'
        object_position=f'({object_poses[i,0,3]:+.2f}, {object_poses[i,1,3]:+.2f}, {object_poses[i,2,3]:+.2f}) m' if valid[i] else 'non disponibile'
        status.content=f'**Frame {i}/{count-1} · {data["relative_timestamps"][i]:.2f} s**  \nCamera MPS: {"valida" if camera_valid[i] else "non disponibile"} · sedia: {object_position}  \nMetodo: {method}; {quality}'

    frame.on_update(update);show_paths.on_update(update);follow.on_update(update)
    update()
    print(f'Viewer ready: http://{args.host}:{args.port}',flush=True)
    next_frame_at=time.monotonic()
    try:
        while True:
            now=time.monotonic()
            if play.value and now>=next_frame_at:
                frame.value=(int(frame.value)+1)%count
                next_frame_at=now+1/fps.value
            elif not play.value:
                next_frame_at=now
            time.sleep(.03)
    except KeyboardInterrupt:
        server.stop()


if __name__=='__main__':main()
