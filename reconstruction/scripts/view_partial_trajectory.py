#!/usr/bin/env python3
"""Preview a snapshot of completed object poses in the OpenCV camera frame.

This does not compose unverified object translation units with metric MPS
translations. It never changes tracking outputs and allocates no CUDA tensors.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser


A = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
D = np.diag([-1., -1., 1.])


def snapshot(sequence, output):
    import torch
    completed = {}
    skipped = []
    for path in sorted((sequence / 'obj_tracking_out/whitechair').glob('frame_*_samples.pt')):
        try:
            saved = torch.load(path, map_location='cpu', weights_only=False)
        except (OSError, RuntimeError, EOFError) as exc:
            skipped.append({'path': str(path), 'reason': str(exc)})
            continue
        selected = next(x for x in saved['samples'] if x['sample_seed'] == saved['best_seed'])
        completed[saved['frame_idx']] = selected
    init = 28
    layout = json.loads((sequence / f'video_segmentation/masks/frame_{init:06d}_masks/layout.json').read_text())['objects'][0]['local_to_scene']
    completed[init] = {'translation': np.array(layout['translation']), 'rotation': np.array(layout['quat_wxyz']), 'scale': np.array(layout['scale'])}
    n = max(completed) + 1
    timestamps = np.load(sequence / 'timestamps_ns.npy')[:n]
    data = {'frame_indices': np.arange(n), 'timestamps_ns': timestamps, 'relative_timestamps': (timestamps - timestamps[0]) * 1e-9, 'valid': np.zeros(n, dtype=bool), 'fixed_scale': np.array(layout['scale']), 'metric_scale_verified': np.array(False), 'translation_units': np.array('unverified SAM3D/MoGe units'), 'coordinate_frame': np.array('camera: x-right,y-down,z-forward')}
    for kind in ('raw', 'post'):
        data[kind + '_translations'] = np.full((n, 3), np.nan)
        data[kind + '_rotations'] = np.full((n, 3, 3), np.nan)
        data[kind + '_quaternions_wxyz'] = np.full((n, 4), np.nan)
    for i, selected in completed.items():
        for kind in ('raw', 'post'):
            prefix = 'post_opt_' if kind == 'post' and 'post_opt_rotation' in selected else ''
            def array(key):
                value = selected[prefix + key]
                return value.detach().float().reshape(-1).numpy() if hasattr(value, 'detach') else np.asarray(value).reshape(-1)
            quaternion, translation, scale = array('rotation'), array('translation'), array('scale')
            if not (np.isfinite(quaternion).all() and np.isfinite(translation).all()):
                raise ValueError(f'Nonfinite {kind} pose at frame {i}')
            if not np.allclose(scale, data['fixed_scale'], atol=1e-5):
                raise ValueError(f'Scale differs from reference at frame {i}')
            rotation_raw = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            rotation_camera = D @ rotation_raw.T @ A
            translation_camera = D @ translation
            # Validate the row-vector -> column-vector conversion on OBJ points.
            probe = np.array([[.1, -.2, .3], [-.3, .4, .2]])
            scale_obj = A.T @ np.diag(scale) @ A
            direct = (((probe @ A.T) * scale) @ rotation_raw + translation) @ D.T
            converted = (probe @ scale_obj.T) @ rotation_camera.T + translation_camera
            if not np.allclose(direct, converted, atol=1e-6):
                raise AssertionError('Pose convention round trip failed')
            data[kind + '_translations'][i] = translation_camera
            data[kind + '_rotations'][i] = rotation_camera
            data[kind + '_quaternions_wxyz'][i] = Rotation.from_matrix(rotation_camera).as_quat()[[3, 0, 1, 2]]
        data['valid'][i] = True
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'poses_camera.npz', **data)
    summary = {'created_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'last_frame': n - 1, 'valid_pose_frames': int(data['valid'].sum()), 'missing_frames': np.flatnonzero(~data['valid']).tolist(), 'duration_s': float(data['relative_timestamps'][-1]), 'coordinate_frame': str(data['coordinate_frame']), 'translation_units': str(data['translation_units']), 'metric_scale_verified': False, 'skipped_files': skipped}
    (output / 'snapshot.json').write_text(json.dumps(summary, indent=2) + '\n')
    return data, summary


def save_plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 5), constrained_layout=True)
    for j, (kind, title) in enumerate([('raw', 'Posa grezza'), ('post', 'Posa ottimizzata')]):
        ax = fig.add_subplot(1, 2, j + 1, projection='3d')
        points = data[kind + '_translations']
        valid = data['valid']
        # NaN explicitly breaks the trajectory across unavailable frames.
        ax.plot(*points.T, color='#999999', alpha=.65, lw=1)
        scatter = ax.scatter(*points[valid].T, c=data['relative_timestamps'][valid], cmap='viridis', s=16)
        ax.scatter(0, 0, 0, color='black', marker='^', s=55, label='Camera')
        ax.scatter(*points[28], color='red', marker='*', s=95, label='Riferimento: frame 28')
        ax.set(title=title, xlabel='X: destra', ylabel='Y: basso', zlabel='Z: avanti')
        ax.legend(fontsize=8)
        ax.view_init(elev=20, azim=-65)
        fig.colorbar(scatter, ax=ax, shrink=.6, label='Tempo originale (s)')
    fig.suptitle('Traiettoria parziale relativa alla camera — unità non verificate in metri', fontsize=12)
    fig.savefig(output / 'trajectory_camera.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    sequence, output = args.sequence.resolve(), args.output.resolve()
    data, summary = snapshot(sequence, output)
    save_plot(data, output)
    print(json.dumps(summary, indent=2), flush=True)
    if args.snapshot_only:
        return
    metadata = json.loads((sequence / 'aria_metadata.json').read_text())
    mesh = trimesh.load(sequence / 'video_segmentation/masks/frame_000028_masks/whitechair/whitechair.obj', force='mesh')
    scale_obj = A.T @ np.diag(data['fixed_scale']) @ A
    mesh.vertices = np.asarray(mesh.vertices) @ scale_obj.T
    server = viser.ViserServer(host=args.host, port=args.port, label='Room4 — traiettoria white chair')
    server.scene.set_up_direction('-y')
    server.gui.add_markdown('**Traiettoria parziale relativa alla camera**\n\nInclude il movimento della testa. Le unità della traslazione non sono ancora verificate in metri.\n\nBlu: posa ottimizzata. Arancione: posa grezza. Il frame 27 non ha una posa.')
    frame = server.gui.add_slider('Frame originale', min=0, max=summary['last_frame'], step=1, initial_value=28)
    mode = server.gui.add_dropdown('Posa della mesh', ['Ottimizzata', 'Grezza'], initial_value='Ottimizzata')
    play = server.gui.add_checkbox('Riproduci', False)
    fps = server.gui.add_slider('Frame al secondo', min=1, max=20, step=1, initial_value=10)
    show_paths = server.gui.add_checkbox('Mostra traiettorie', True)
    image = server.gui.add_image(np.zeros((288, 384, 3), dtype=np.uint8), label='Immagine e maschera SAM3', format='jpeg')
    status = server.gui.add_markdown('')
    paths = []
    for kind, color in [('post', (40, 130, 255)), ('raw', (255, 150, 40))]:
        points = data[kind + '_translations']
        adjacent = data['valid'][:-1] & data['valid'][1:]
        segments = np.stack([points[:-1][adjacent], points[1:][adjacent]], axis=1)
        paths.append(server.scene.add_line_segments('/trajectory/' + kind, segments, np.array(color, dtype=np.uint8), line_width=3))
        paths.append(server.scene.add_point_cloud('/trajectory/' + kind + '_points', points[data['valid']], color, point_size=.009))
    object_handle = server.scene.add_mesh_trimesh('/chair', mesh)
    object_axes = server.scene.add_frame('/chair_axes', axes_length=.16, axes_radius=.004)
    server.scene.add_frame('/camera_axes', axes_length=.24, axes_radius=.004)
    height, width = metadata['pinhole_size'][1], metadata['pinhole_size'][0]
    server.scene.add_camera_frustum('/camera', fov=2 * np.arctan(height / (2 * metadata['focal_px'])), aspect=width / height, scale=.16, wxyz=(0., 1., 0., 0.), color=(70, 70, 70))
    positions = data['post_translations'][data['valid']]
    center = positions.mean(axis=0)

    @server.on_client_connect
    def setup_camera(client):
        client.camera.up_direction = (0., -1., 0.)
        client.camera.position = tuple(center + np.array([1.3, -.9, -1.8]))
        client.camera.look_at = tuple(center)

    def update(_=None):
        i = int(frame.value)
        kind = 'post' if mode.value == 'Ottimizzata' else 'raw'
        for path in paths:
            path.visible = show_paths.value
        img = cv2.imread(str(sequence / f'all_frames/{i:06d}.png'))
        mask = cv2.imread(str(sequence / f'video_segmentation/masks/frame_{i:06d}_masks/whitechair.png'), 0)
        if mask is not None:
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (20, 255, 20), 2)
        image.image = cv2.cvtColor(cv2.resize(img, (384, 288)), cv2.COLOR_BGR2RGB)
        visible = bool(data['valid'][i])
        object_handle.visible = object_axes.visible = visible
        if visible:
            rotation = data[kind + '_quaternions_wxyz'][i]
            translation = data[kind + '_translations'][i]
            with server.atomic():
                object_handle.wxyz = object_axes.wxyz = rotation
                object_handle.position = object_axes.position = translation
            status.content = f'**Frame {i} · {data["relative_timestamps"][i]:.2f} s**\n\n{mode.value}: posizione rispetto alla camera = ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}), in unità non verificate.\n\nSnapshot fino al frame {summary["last_frame"]} ({summary["duration_s"]:.2f} s).'
        else:
            status.content = f'**Frame {i} · {data["relative_timestamps"][i]:.2f} s**\n\nPosa non disponibile: mesh nascosta. Nessuna interpolazione.'

    frame.on_update(update)
    mode.on_update(update)
    show_paths.on_update(update)
    update()
    print(f'Viewer ready: http://{args.host}:{args.port}', flush=True)
    next_frame_at = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if play.value and now >= next_frame_at:
                frame.value = (int(frame.value) + 1) % (summary['last_frame'] + 1)
                next_frame_at = now + 1. / fps.value
            elif not play.value:
                next_frame_at = now
            time.sleep(.03)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
