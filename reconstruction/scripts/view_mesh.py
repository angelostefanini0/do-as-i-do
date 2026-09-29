#!/usr/bin/env python3
"""View a textured reference mesh in a browser, independently of tracking."""

import argparse
from pathlib import Path
import time

import numpy as np
import trimesh
import viser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    mesh_path = args.mesh.resolve(strict=True)
    mesh = trimesh.load(mesh_path, force='mesh')
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f'Mesh has no geometry: {mesh_path}')

    server = viser.ViserServer(host=args.host, port=args.port, label=mesh_path.stem)
    server.scene.set_up_direction('+y')
    server.scene.add_mesh_trimesh('/object', mesh)
    server.gui.add_markdown(
        f'### {mesh_path.stem}\n\n'
        'Trascina per ruotare; usa la rotella per lo zoom.'
    )
    center = np.asarray(mesh.bounds).mean(axis=0)
    distance = max(float(np.max(mesh.extents)), 0.01)

    @server.on_client_connect
    def set_camera(client):
        client.camera.up_direction = (0.0, 1.0, 0.0)
        client.camera.position = tuple(center + distance * np.array([1.5, 1.0, 1.5]))
        client.camera.look_at = tuple(center)

    print(f'Viewing {mesh_path} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
