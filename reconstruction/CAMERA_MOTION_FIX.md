# White chair in Room4: camera-compensated reconstruction

Run `./run_camera_aware_room4.sh room4` from this directory. It reuses the
extracted Aria images, SAM3 masks, reference SAM3D chair mesh, MPS camera poses,
MPS semidense points, and local BootsTAPIR weights. It does not use HaWoR or
the camera-relative Fast-SAM3D pose chain. `STATIC_FRAMES=20:56` identifies the
visually verified stationary setup interval; it is used only to calibrate the
initial metric chair geometry and landmarks. Later chair motion is estimated
freely. `TAPIR_GPU=1` changes the GPU used for the short TAPIR stage.

The estimated pose uses the factory-calibrated RGB pinhole intrinsics
(`fx=fy=334.4843518808806 px` at 768×576), not MoGe's inferred focal length.
MPS supplies an exact timestamp-aligned, metric camera-to-world pose. For each
frame the previous chair pose is transferred into the **new** moving camera by
`inv(T_world_camera_now) @ T_world_camera_previous @ T_camera_object_previous`.
Tracked chair landmarks then yield a new pose through robust PnP. If visual
measurements fail, the mesh is fitted to the SAM3 mask for a short local gap
and labelled as a lower-confidence recovery. Persistent loss remains invalid.
The rigid chair mesh scale is fitted to MPS chair surface points and calibrated
silhouettes; no per-frame mesh reconstruction is implied.

## Validated Room4 result

- 307 original RGB frame indices and timestamps are preserved; 250 have a
  camera-compensated **world** chair pose. The camera MPS trajectory is absent
  for frames 0–11 and 306. Missing chair measurements remain `NaN`, never a
  copied pose or interpolated trajectory.
- While the chair is stationary in frames 20–55, its world position differs
  from frame 28 by a median 0.73 cm (90th percentile 3.67 cm), despite camera
  travel of roughly 0.66 m across the calibration views. Median world rotation
  step is 0.34 degrees. The old camera-relative result had steps up to about
  31 degrees in the same stationary segment.
- The held-out initial mesh silhouette IoU with real Aria intrinsics is 0.851
  median. Across all valid frames it is 0.562 median; this lower score reflects
  perspective, occlusion by hands, and imperfections of the fixed SAM3D mesh.
  The world trajectory reaches 1.93 m from the initial chair pose while the
  chair is visibly carried; motion is not artificially frozen.
- SAM3 mistakenly switches to the **gray chair** at frames 260–272 after the
  white chair leaves the image. Those frames are rejected using appearance,
  along with nonphysical jumps. Frames 275–280 are independently recovered by
  tracked features when the white chair briefly returns. Later small,
  phone-occluded views lack a reliable 3D re-identification and are left invalid.

The mesh is an approximate visual model, not a watertight collision asset or a
guarantee of millimeter-accurate pose during occlusion. Confidence and method
are per frame; downstream ECHO use must respect `world_valid` and must not
silently treat `NaN` as a held chair pose.

## Files and coordinate convention

- [`room4/camera_motion_fix/calibrated/room4_whitechair_trajectory_mps.npz`](room4/camera_motion_fix/calibrated/room4_whitechair_trajectory_mps.npz): `T_world_object`, `T_camera_object`, `T_world_camera`, frame/timestamp arrays,
  `valid`, `world_valid`, `confidence`, `method`, calibrated `K`, and mesh path.
  All transforms use column vectors; translations are meters. The MPS world is
  `+z` up, while Aria RGB camera coordinates are `+x` right, `+y` down, `+z`
  forward. `T_world_object = T_world_camera @ T_camera_object` when valid.
- [`room4/camera_motion_fix/whitechair_metric.obj`](room4/camera_motion_fix/whitechair_metric.obj): one reference SAM3D mesh with its metric scale baked in. Apply a valid `T_world_object` to place it in the room.
- [`room4/camera_motion_fix/calibrated/metrics.json`](room4/camera_motion_fix/calibrated/metrics.json) and [`frame_metrics.csv`](room4/camera_motion_fix/calibrated/frame_metrics.csv): coverage, geometric checks,
  rejection reasons, and per-frame reprojection/silhouette quality.
- [`room4/camera_motion_fix/calibrated/room4_whitechair_world_preview.mp4`](room4/camera_motion_fix/calibrated/room4_whitechair_world_preview.mp4): side-by-side RGB mesh overlay and fixed-scale MPS world map with camera and chair trajectories.

The interactive world viewer runs at `http://127.0.0.1:8081` from this host.
It shows the camera frustum moving relative to the chair, the calibrated mesh,
reference MPS chair points, per-frame method/confidence, RGB contours, and
invalid gaps. Re-launch with:

```bash
micromamba run -n sam3d python scripts/view_world_trajectory.py \
  --sequence room4 \
  --trajectory room4/camera_motion_fix/calibrated/room4_whitechair_trajectory_mps.npz \
  --port 8081
```

The runner skips existing upstream artifacts to permit resuming after an
interruption, then recomputes the corrected trajectory. Removing a particular
upstream artifact causes only that stage to run again. The legacy
`room4/resume_pipeline.sh` belongs to the uncompensated run and should not be
used for the corrected trajectory.
