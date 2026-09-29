#!/usr/bin/env bash
# Reconstruct the white chair in metric Aria MPS world coordinates from cached
# room4 RGB frames, masks, SAM3D mesh, and local model weights. No HaWoR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA="${MICROMAMBA_BIN:-/home/astefanini/bin/micromamba}"
SEQUENCE="$(realpath "${1:-$HERE/room4}")"
OUT="$SEQUENCE/camera_motion_fix"
VRS="$($MAMBA run -n sam3d python - "$SEQUENCE/aria_metadata.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['vrs'])
PY
)"
STEM="$(basename "$VRS" .vrs)"
MPS_POINTS="$(dirname "$VRS")/mps_${STEM}_vrs/slam/semidense_points.csv.gz"
STATIC_FRAMES="${STATIC_FRAMES:-20:56}"
REFERENCE_FRAME="${REFERENCE_FRAME:-28}"
TAPIR_GPU="${TAPIR_GPU:-1}"
[[ -f "$MPS_POINTS" && -f "$SEQUENCE/aria_camera_trajectory.npz" && -f "$SEQUENCE/video_segmentation/masks/frame_$(printf '%06d' "$REFERENCE_FRAME")_masks/whitechair/whitechair.obj" ]] || {
  echo 'Missing MPS points/camera poses or reference white-chair mesh' >&2; exit 1;
}
mkdir -p "$OUT" "$OUT/calibrated"
if [[ ! -f "$OUT/landmarks.npz" ]]; then
  "$MAMBA" run -n sam3d python "$HERE/scripts/prepare_mps_object_landmarks.py" \
    --sequence "$SEQUENCE" --mps-points "$MPS_POINTS" --reference "$REFERENCE_FRAME" \
    --static-frames "$STATIC_FRAMES" --output "$OUT"
fi
if [[ ! -f "$OUT/landmark_tracks.npz" ]]; then
  CUDA_VISIBLE_DEVICES="$TAPIR_GPU" "$MAMBA" run -n tapnet python "$HERE/scripts/track_mps_landmarks.py" \
    --sequence "$SEQUENCE" --landmarks "$OUT/landmarks.npz" \
    --checkpoint "$HERE/weights/tapnet/bootstapir_checkpoint_v2.pt" \
    --output "$OUT/landmark_tracks.npz"
fi
if [[ ! -f "$OUT/landmarks_triangulated.npz" ]]; then
  "$MAMBA" run -n sam3d python "$HERE/scripts/triangulate_rigid_landmarks.py" \
    --sequence "$SEQUENCE" --landmarks "$OUT/landmarks.npz" \
    --tracks "$OUT/landmark_tracks.npz" --static-frames "$STATIC_FRAMES" \
    --output "$OUT/landmarks_triangulated.npz"
fi
if [[ ! -f "$OUT/mesh_anchor.npz" ]]; then
  "$MAMBA" run -n sam3d python "$HERE/scripts/fit_mesh_to_mps.py" \
    --sequence "$SEQUENCE" --landmarks "$OUT/landmarks_triangulated.npz" --output "$OUT"
fi
"$MAMBA" run -n sam3d python -u "$HERE/scripts/track_calibrated_object.py" \
  --sequence "$SEQUENCE" --landmarks "$OUT/landmarks_triangulated.npz" \
  --tracks "$OUT/landmark_tracks.npz" --anchor "$OUT/mesh_anchor.npz" \
  --output "$OUT/calibrated"
echo "Metric camera-compensated trajectory: $OUT/calibrated/room4_whitechair_trajectory_mps.npz"
