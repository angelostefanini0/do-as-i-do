# Room4 object-only reconstruction

**Update 2026-09-29:** The original Fast-SAM3D pose chain described below was
camera-relative and does not compensate Aria head motion. The corrected Room4
white-chair runner is [`run_camera_aware_room4.sh`](run_camera_aware_room4.sh),
with results and validation in [`CAMERA_MOTION_FIX.md`](CAMERA_MOTION_FIX.md).
Use its metric `camera_motion_fix/calibrated` output for visualization or ECHO;
the older `obj_tracking_out` samples are retained only for comparison. A fresh
`run_pipeline_no_hawor.sh` run on this exact Room4 VRS now delegates to the
camera-aware runner after creating the reference mesh. The old pose chain is
available only with `--legacy-camera-relative`.

The entry point is `run_pipeline_no_hawor.sh`. It uses `micromamba run` in the
existing `sam3`, `sam3d`, and `tapnet` environments. VRS decoding uses the already
installed Project Aria Tools in `/home/astefanini/egoallo/.mamba-env` (override
`ARIA_ENV_PREFIX` if needed). It never invokes environment setup, installs, HaWoR,
MANO, hand segmentation, or `optimize_translation_scale.py`. Hugging Face access
is offline so missing weights fail instead of starting downloads.

Detected inputs:

```
/home/astefanini/AriaData/Video_moving_around_room_4_20260928_232041.vrs
/home/astefanini/AriaData/mps_Video_moving_around_room_4_20260928_232041_vrs/slam/closed_loop_trajectory.csv
```

The VRS has 307 RGB images (2560 x 1920), with about 30.6 seconds between the
first and last exposure timestamps. Frame 28 shows the white chair before it is
lifted. The prepared images are undistorted pinhole images, 768 x 576, in the
native RGB orientation. There is no additional image rotation. The factory RGB
extrinsic therefore applies directly to this virtual camera. The pinhole focal
length is the factory focal length scaled by 768/2560 (about 334.48 pixels).
`aria_metadata.json` records the actual preparation parameters.

Run from this directory, after pointing at the downloaded weights if necessary:

```bash
./run_pipeline_no_hawor.sh \
  /home/astefanini/AriaData/Video_moving_around_room_4_20260928_232041.vrs \
  28 whitechair
```

Add `--dry-run` to print every stage without executing it. The default SAM3 prompt
is `white chair`, with output ID `whitechair`. Add `--click` for manual target
selection in the reference frame. SAM3 propagates both forward and backward.
`SAM3_CHECKPOINT=/absolute/path/sam3.pt` overrides its cache; the existing
segmentation script now actually forwards that argument to its model builder.
`SAM3D_CONFIG=/absolute/path/to/downloaded/pipeline.yaml` supplies the existing
SAM3D weights/configs to both reconstruction and tracking. Its referenced
checkpoint/config files must exist relative to that YAML, or use absolute paths.
The checked-in Fast-SAM3D YAML is usable only when its heavy checkpoints are also
present. The readiness check on 2026-09-29 confirmed the required checkpoints
and successfully ran chair mesh generation, three-frame tracking, and NPZ
export. The missing DINOv2 Torch Hub code/weights cache was populated during
that check. See [ROOM4_READINESS.md](ROOM4_READINESS.md) for evidence, GPU memory
measurements, and the limits of this short test.

Kept stages are VRS RGB extraction and undistortion, lossless MP4 encoding, SAM3
object masks, one SAM3D reference mesh, reference/all-frame MoGe, GeoCalib, TAPIR,
Fast-SAM3D tracking, camera conversion, and NPZ export. All are independent of
HaWoR in this mode. GeoCalib saves per-frame camera-coordinate gravity vectors;
the static-camera average is disabled. TAPIR supplies apparent image motion to
the existing adaptive pose guidance; it does not estimate a world trajectory.
Fast-SAM3D internally recomputes MoGe for tracking: the standalone pointmap files
and GeoCalib file are retained outputs, not inputs to this tracker.

Outputs below `reconstruction/room4`:

```
all_frames/000000.png ... 000306.png
room4.mp4
timestamps_ns.npy
aria_metadata.json
aria_camera_trajectory.npz
video_segmentation/masks/frame_000028_masks/whitechair/whitechair.obj
whitechair_fixed.obj
obj_tracking_out/whitechair/combined_visualization/layout.json
obj_tracking_out/whitechair/combined_visualization/layout_camera_frame.json
room4_whitechair_trajectory.npz
```

Tracking voxelizes and reuses the same reference mesh for every frame, with
`--fix_scale_to_init_frame`. It does not reconstruct a new chair each frame.
Combined layout `mesh_obj` names such as `whitechair_frame28.obj` are labels;
`save_combined_visualization` does not create those per-frame OBJ files. Use the
single reference OBJ, or the exported `whitechair_fixed.obj` with its fixed scale
baked in. The runner refuses to mix a VRS extraction with an existing nonempty
`all_frames` directory; use a fresh `--output-dir` for another full run.

## Coordinate systems, derived from the actual source

`track_object.py:compose_transform` calls PyTorch3D's
`scale(...).rotate(quaternion_to_matrix(q)).translate(t)`. Those transforms act
on **row** vectors. The exported canonical OBJ is y-up; the tracker first changes
it to its z-up object convention. Let `R_raw = quaternion_to_matrix(quat_wxyz)`
and `s = local_to_scene.scale`. Using **column** vectors:

```text
A = [[1,0,0], [0,0,-1], [0,1,0]]       # OBJ -> internal z-up object
D = diag(-1,-1,1)                      # PyTorch3D -> OpenCV camera
x_p3d = R_raw.T @ diag(s) @ A @ x_obj + t_raw
x_camera = D @ x_p3d
```

`layout.json` has `frame: per_camera`. Each entry belongs to its own time-varying
camera, not a shared world. Its fields use two conventions:

* `translation` is raw PyTorch3D camera translation: x-left, y-up, z-forward.
* `quat_wxyz` (also `quat_xyzw`) is the raw internal rotation used on row vectors.
  It needs a transpose when forming a conventional column-vector transform.
* `new_quat` is a column-vector rotation fitted by Procrustes from the exported
  OBJ into the Isaac-style pose axes: x-forward, y-left, z-up. That scene origin
  is still the current camera. Its matching translation is `[tz, tx, ty]`, not
  the unchanged raw `translation` field.
* `scale` is part of the mesh placement, separate from its rigid pose. A 4x4
  rigid transform alone does not place the unscaled reference OBJ correctly.

`convert_layout_to_camera_frame.py` preserves the above fields and adds
`translation_camera_frame` and `quat_wxyz_camera_frame` in x-right, y-down,
z-forward. It applies:

```text
t_camera = [-tx, -ty, tz]               # at --translation-scale 1
R_camera = P.T @ R(new_quat)
P = [[0,0,1], [-1,0,0], [0,-1,0]]
```

For uniform scale, this is exactly `R_camera = D @ R_raw.T @ A`. For anisotropic
scale, Procrustes `new_quat` can be an approximation. The new exporter derives
the rigid rotation directly from `quat_wxyz` and bakes the fixed scale in OBJ
axes (`A.T @ diag(s) @ A`), avoiding that approximation. It validates that scale
is constant. The export's object origin is the reference OBJ origin, not an
independently measured chair center.

Numerical CPU checks compare these conversions to actual PyTorch3D-transformed
mesh points, including nontrivial rotations and camera-motion composition.

## Camera motion and MPS

Without `--hdf5_path`, chained guidance reuses the previous camera-relative pose
as a prior without accounting for camera motion. Current-frame RGB/pointmap
inference still produces camera-relative estimates, but guidance can be biased
by Aria movement. `--hdf5_path` compensates the prior only; the saved layouts
remain per-camera poses even when camera extrinsics are supplied.

Fast-SAM3D expects `transforms/camera[N,4,4]`, indexed by the **original extracted
frame index**, containing **camera-to-world** column-vector transforms in an
ARKit camera basis (x-right, y-up, z-back). It loads:

```text
F = diag(-1,1,-1,1)
T_world_camera_p3d(t) = HDF5["transforms/camera"][t] @ F
T_camera_dst_object = inv(T_world_camera_p3d(dst))
                      @ T_world_camera_p3d(src) @ T_camera_src_object
```

The small correction in `transform_pose_between_cameras` transposes raw rotation
matrices into and out of this column-vector equation. Translation direction
already followed the equation above. This was checked against the renderer,
including a camera-to-camera-to-original-camera round trip.

The VRS preparation saves MPS-aligned **OpenCV** camera-to-world poses:

```text
T_world_camera(t) = T_world_device(t) @ T_device_rgb
T_world_object(t) = T_world_camera(t) @ T_camera_object(t)
```

MPS `tracking_timestamp_us` and RGB exposure timestamps are aligned in the
device clock, with linear translation interpolation and quaternion SLERP.
Factory camera exposure offset is applied; it is zero for this recording.
MPS positions are in meters. See the official
[MPS trajectory format](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/mps_trajectory)
and [RGB extrinsic composition example](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/mps).
Missing MPS intervals/quality or out-of-range timestamps remain invalid/NaN.
Room4 MPS covers exactly 294 RGB frames: indices 12 through 305 inclusive.
Indices 0 through 11 and 306 have no covered MPS pose. MPS begins at
55.644649 s and ends at 85.044649 s; RGB begins at
54.501501393 s and ends at 85.098274351 s, so it does not cover every RGB frame.
For higher accuracy, use matching per-frame online calibration and handle
rolling shutter; this minimal preparation uses the VRS factory calibration.

**Object metric scale has not been verified.** The configured MoGe v1 model and
SAM3D placement do not establish a calibrated metric object scale. Fixing scale
to the reference frame does not prove meters or remove framewise depth bias.
Verify against measured chair dimensions, calibrated depth, or MPS point-cloud
correspondences before mixing camera and object translations. If the chair is
known stationary, its estimated world pose is also a useful consistency check,
not evidence on its own that the scale is correct.

After verifying `s = meters per object-pose unit`, an HDF5 adapter can be made
from `aria_camera_trajectory.npz`:

```python
import h5py
import numpy as np
c = np.load("room4/aria_camera_trajectory.npz")
H = c["T_world_camera"].copy()
H[:, :3, 3] /= s                         # meters -> object pose units
H = H @ np.diag([1., -1., -1., 1.])     # OpenCV -> ARKit basis
with h5py.File("room4/camera_arkit_pose_units.h5", "w") as f:
    f.create_dataset("transforms/camera", data=H)
```

Use only frames with valid camera poses, preserve the full HDF5 frame indexing,
and pass `--hdf5-path room4/camera_arkit_pose_units.h5 --frames 12:306`. This file
must not be passed at its original meter scale unless the object pose units
have independently been shown to be meters.

## NPZ trajectory

`room4_whitechair_trajectory.npz` opens with `np.load(..., allow_pickle=False)`:

| Key | Shape / meaning |
| --- | --- |
| `timestamps_ns` | N, int64 Aria device exposure timestamps |
| `timestamps`, `relative_timestamps` | N, seconds in device clock / since first frame |
| `frame_indices`, `valid` | N, original indices / pose availability |
| `translations` | N x 3 camera-relative object-origin positions |
| `rotations` | N x 3 x 3 column-vector rotation matrices |
| `quaternions_wxyz` | N x 4 |
| `transforms`, `T_camera_object` | N x 4 x 4 rigid transforms of the fixed mesh |
| `fixed_scale_sam3d`, `mesh_scale_matrix` | The constant scale baked in the exported OBJ |
| `T_world_camera`, `camera_valid` | Separate metric camera poses and their availability |
| `metric_scale_verified`, `translation_units` | Explicit scale status |

Every input frame has a slot; missing/empty masks or skipped poses have
`valid=False` and NaNs, rather than fabricated poses. World-object poses are
omitted until verified scale is explicitly supplied. To export again after
verifying scale, add `--verified-meters-per-unit VALUE`:

```bash
micromamba run -n sam3d python scripts/export_object_trajectory.py \
  --layout room4/obj_tracking_out/whitechair/combined_visualization/layout_camera_frame.json \
  --timestamps-ns room4/timestamps_ns.npy --object whitechair \
  --mesh room4/video_segmentation/masks/frame_000028_masks/whitechair/whitechair.obj \
  --fixed-mesh room4/whitechair_fixed_metric.obj \
  --camera-trajectory room4/aria_camera_trajectory.npz \
  --output room4/room4_whitechair_trajectory_metric.npz \
  --verified-meters-per-unit VALUE
```

That adds `T_world_object` and `world_valid`, and expresses both the fixed mesh
and object translations in meters. A different camera trajectory can use the
same NPZ schema if it has matched timestamps, the correct image-camera basis,
camera-to-world column-vector transforms, and meter translations.
