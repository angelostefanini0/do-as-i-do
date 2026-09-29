#!/usr/bin/env bash
# Object-only reconstruction. Reuses existing environments and model weights.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config/paths.sh"

usage() {
    cat <<'EOF'
Usage: ./run_pipeline_no_hawor.sh INPUT.vrs|INPUT.mp4 [REF_FRAME=28] [OBJECT=whitechair] [options]
  --output-dir DIR        Default: <reconstruction>/room4 for VRS; video directory for MP4
  --mps CSV              Default: sibling mps_<VRS basename>_vrs/slam/closed_loop_trajectory.csv
  --click                Interactive object selection (default: text "white chair")
  --hdf5-path FILE        Optional ARKit camera-to-world transforms/camera[N,4,4],
                         translations MUST already use the SAME units as SAM3D poses
  --frames START:END     Optional tracking range (END exclusive; indices remain original)
  --legacy-camera-relative  Use the older Fast-SAM3D pose chain for Room4
  --dry-run              Print commands and output paths; no extraction or inference
For the already prepared Room4 sequence, run ./run_camera_aware_room4.sh room4.
Environment overrides: ARIA_ENV_PREFIX, SAM3_CHECKPOINT, SAM3D_CONFIG,
  SAM3_TEXT, MICROMAMBA_BIN. No setup, dependency installs, or downloads.
EOF
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# -gt 0 ]] || { usage; exit 2; }
INPUT="$(realpath "$1")"; shift
REF=28; OBJECT=whitechair
if [[ $# -gt 0 && $1 != --* ]]; then REF=$1; shift; fi
if [[ $# -gt 0 && $1 != --* ]]; then OBJECT=$1; shift; fi
[[ $REF =~ ^[0-9]+$ && $OBJECT =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid frame or object ID" >&2; exit 2; }
REF=$((10#$REF))
OUT=""; MPS=""; HDF5=""; FRAMES=""; stem=""; DRY=0; CLICK=0; LEGACY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUT=$2; shift 2;;
        --mps) MPS=$2; shift 2;;
        --hdf5-path) HDF5=$2; shift 2;;
        --frames) FRAMES=$2; shift 2;;
        --dry-run) DRY=1; shift;;
        --click) CLICK=1; shift;;
        --legacy-camera-relative) LEGACY=1; shift;;
        *) echo "Unknown option: $1" >&2; usage; exit 2;;
    esac
done
MAMBA="${MICROMAMBA_BIN:-micromamba}"
ARIA_PREFIX="${ARIA_ENV_PREFIX:-/home/astefanini/egoallo/.mamba-env}"
CONFIG="$(realpath -m "${SAM3D_CONFIG:-$FASTSAM3D_DIR/checkpoints/hf/pipeline.yaml}")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run() { printf '+ '; printf '%q ' "$@"; printf '\n'; if (( ! DRY )); then "$@"; fi; }
in_env() { local env_name=$1; shift; run "$MAMBA" run -n "$env_name" "$@"; }
if [[ $INPUT == *.vrs ]]; then
    OUT="$(realpath -m "${OUT:-$HERE/room4}")"
    stem="$(basename "$INPUT" .vrs)"
    MPS="${MPS:-$(dirname "$INPUT")/mps_${stem}_vrs/slam/closed_loop_trajectory.csv}"
    VIDEO="$OUT/room4.mp4"
else
    OUT="$(realpath -m "${OUT:-$(dirname "$INPUT")}")"
    [[ $OUT == "$(dirname "$INPUT")" ]] || { echo "For MP4, outputs must use its parent directory (existing stage convention)" >&2; exit 2; }
    VIDEO=$INPUT
fi
REF_IMAGE="$OUT/$(printf '%04d.png' "$REF")"
MASKS="$OUT/video_segmentation/masks"
REF_MASKS="$MASKS/$(printf 'frame_%06d_masks' "$REF")"
MESH="$REF_MASKS/$OBJECT/$OBJECT.obj"
TRACK="$OUT/obj_tracking_out/$OBJECT"
COMBINED="$TRACK/combined_visualization"
TRAJECTORY="$OUT/room4_${OBJECT}_trajectory.npz"
FIXED_MESH="$OUT/${OBJECT}_fixed.obj"
printf 'Input: %s\nVideo: %s\nReference frame: %s\nMesh: %s\nFixed scale-baked mesh: %s\nLayout: %s/layout.json\nCamera layout: %s/layout_camera_frame.json\nTrajectory: %s\n' \
    "$INPUT" "$VIDEO" "$REF" "$MESH" "$FIXED_MESH" "$COMBINED" "$COMBINED" "$TRAJECTORY"

# Fail before expensive inference if local weights/configs are incomplete.
if (( ! DRY )); then
    [[ -f $INPUT ]] || { echo "Missing input: $INPUT" >&2; exit 1; }
    "$MAMBA" run -n "$ENV_SAM3D" python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
p = Path(sys.argv[1]); cfg = OmegaConf.load(p)
missing = [str(p.parent / v) for k,v in cfg.items()
           if (k.endswith('_ckpt_path') or k.endswith('_config_path')) and v
           and not (p.parent / v).is_file()]
if missing:
    raise SystemExit('Missing existing SAM3D weight/config files:\n' + '\n'.join(missing)
                     + '\nSet SAM3D_CONFIG to your downloaded pipeline.yaml; nothing will be downloaded.')
PY
    [[ -f $TAPNET_CKPT ]] || { echo "Missing TAPIR checkpoint: $TAPNET_CKPT" >&2; exit 1; }
    if [[ -z ${SAM3_CHECKPOINT:-} ]]; then
        SAM3_CHECKPOINT="$("$MAMBA" run -n "$ENV_SAM3" python - <<'PY'
from huggingface_hub import hf_hub_download
try:
    print(hf_hub_download('facebook/sam3', 'sam3.pt', local_files_only=True))
except Exception:
    raise SystemExit('SAM3 weights not found in local cache; set SAM3_CHECKPOINT to the downloaded sam3.pt')
PY
        )"
    fi
    [[ -f $SAM3_CHECKPOINT ]] || { echo "Missing SAM3 checkpoint: $SAM3_CHECKPOINT" >&2; exit 1; }
fi
if [[ $INPUT == *.vrs ]]; then
    run mkdir -p "$OUT"
    run "$MAMBA" run -p "$ARIA_PREFIX" python "$SCRIPTS_DIR/prepare_aria_sequence.py" \
        --vrs "$INPUT" --output-dir "$OUT" --mps "$MPS"
    # One encoded frame per extracted frame. Device timestamps remain authoritative.
    in_env "$ENV_SAM3" ffmpeg -nostdin -n -framerate 10 -i "$OUT/all_frames/%06d.png" \
        -c:v libx264 -crf 0 -pix_fmt yuv444p "$VIDEO"
else
    [[ -f $OUT/timestamps_ns.npy || $DRY == 1 ]] || { echo "Supply timestamps_ns.npy aligned to this Aria MP4 first" >&2; exit 1; }
    run mkdir -p "$OUT/all_frames"
    in_env "$ENV_SAM3" ffmpeg -nostdin -n -i "$VIDEO" -vsync 0 -start_number 0 "$OUT/all_frames/%06d.png"
fi
run cp "$OUT/all_frames/$(printf '%06d.png' "$REF")" "$REF_IMAGE"
if (( ! DRY )); then
    "$MAMBA" run -n "$ENV_SAM3" python - "$OUT/config.json" "$REF" "$OBJECT" <<'PY'
import json, sys
with open(sys.argv[1], 'w') as f:
    json.dump({'frame_number':int(sys.argv[2]), 'object_names':[sys.argv[3]], 'hands':False}, f, indent=2)
PY
fi
SEG=(--text "${SAM3_TEXT:-white chair}")
if (( CLICK )); then SEG=(--click); fi
if [[ -n ${SAM3_CHECKPOINT:-} ]]; then SEG+=(--checkpoint "$SAM3_CHECKPOINT"); fi
in_env "$ENV_SAM3" env DISPLAY="${DISPLAY:-$SAM3_DISPLAY}" python "$SCRIPTS_DIR/run_sam3_video.py" \
    --video "$VIDEO" --obj_id "$OBJECT" --frame_idx "$REF" "${SEG[@]}"
cd "$SAM3D_DIR"
in_env "$ENV_SAM3D" python generate_mesh_sam3d.py --config "$CONFIG" \
    --image_path "$REF_IMAGE" --masks_dir "$REF_MASKS"
cd "$SCRIPTS_DIR"
# This exact VRS has verified static calibration views and metric MPS points.
# Route its default object reconstruction through the corrected camera-aware
# tracker, so a fresh run cannot silently repeat the ego-motion artifact.
if [[ $INPUT == *.vrs && $stem == Video_moving_around_room_4_20260928_232041 \
      && $OBJECT == whitechair && $REF -eq 28 && $LEGACY -eq 0 ]]; then
    [[ -z $FRAMES && -z $HDF5 ]] || {
        echo "The metric Room4 runner uses the full timestamped sequence; --frames/--hdf5-path require --legacy-camera-relative." >&2
        exit 2
    }
    run bash "$HERE/run_camera_aware_room4.sh" "$OUT"
    echo "Room4 white-chair metric camera-aware pipeline complete."
    exit 0
fi
# MoGe/GeoCalib products are retained. Tracking computes its own per-frame MoGe
# pointmaps; the standalone cached files are not consumed by track_object.py.
in_env "$ENV_SAM3D" python get_pointmap_dir.py --image "$REF_IMAGE" --output "${REF_IMAGE%.png}_pointmap.npy"
in_env "$ENV_SAM3D" python get_pointmap_dir.py --image_dir "$OUT/all_frames"
in_env "$ENV_SAM3D" python predict_video_gravity.py "$OUT/all_frames" --moving-camera --output_path "$OUT/gravity.json"
in_env "$ENV_TAPNET" python tapir_velocity_tracking.py --video "$VIDEO" --mask-dir "$MASKS" \
    --object "$OBJECT" --checkpoint "$TAPNET_CKPT" --fps 10
EXTRA=()
if [[ -n $HDF5 ]]; then EXTRA+=(--hdf5_path "$(realpath -m "$HDF5")"); fi
if [[ -n $FRAMES ]]; then EXTRA+=(--frames "$FRAMES"); fi
if [[ -n $HDF5 && $DRY == 0 ]]; then
    # Validate all requested extrinsics before starting GPU tracking.
    "$MAMBA" run -n "$ENV_SAM3D" python - "$HDF5" "$OUT/timestamps_ns.npy" "$FRAMES" <<'PY'
import sys
import h5py
import numpy as np
with h5py.File(sys.argv[1]) as f:
    T = f['transforms/camera'][:]
n = len(np.load(sys.argv[2], allow_pickle=False))
start, end = map(int, sys.argv[3].split(':')) if sys.argv[3] else (0, n)
if T.shape != (n, 4, 4) or not 0 <= start < end <= n:
    raise SystemExit('HDF5 must preserve all original frame indices; check shape/range')
T = T[start:end]
if (not np.isfinite(T).all() or not np.allclose(T[:,3,:], [0,0,0,1])
    or not np.allclose(T[:,:3,:3].transpose(0,2,1) @ T[:,:3,:3], np.eye(3), atol=1e-5)
    or not np.allclose(np.linalg.det(T[:,:3,:3]), 1, atol=1e-5)):
    raise SystemExit('Requested tracking frames include missing or invalid camera transforms')
PY
fi
if [[ -z $HDF5 ]]; then
    echo "Camera-relative tracking: chained guidance has no explicit camera-motion compensation. Scale is unverified."
fi
cd "$FASTSAM3D_DIR"
in_env "$ENV_SAM3D" python track_object.py --config "$CONFIG" --vid_dir "$OUT" --masks_root "$MASKS" \
    --object_name "$OBJECT" --mesh "$MESH" --init_frame "$REF" --output_dir "$TRACK" \
    --guidance_strength 1 --save_layout --fix_scale_to_init_frame --pose_guidance_strength 0.5 \
    --num_pose_samples 25 --scoring_metric render_iou --pose_selection cluster \
    --cluster_dist_thresh 0.3 --cluster_min_size 3 --cluster_w_rot 1.5 --chain_poses \
    --post_optimize --no-enable_shape_icp --chain_on_diffusion --enable_ss_cache \
    --torch_compile --euler_steps 25 --rotvel_json "$OUT/perframe_tracking_$OBJECT/motion_stats.json" "${EXTRA[@]}"
cd "$SCRIPTS_DIR"
in_env "$ENV_SAM3D" python convert_layout_to_camera_frame.py \
    --input "$COMBINED/layout.json" --output "$COMBINED/layout_camera_frame.json"
CAMERA=()
if [[ $INPUT == *.vrs || -f $OUT/aria_camera_trajectory.npz ]]; then
    CAMERA=(--camera-trajectory "$OUT/aria_camera_trajectory.npz")
fi
in_env "$ENV_SAM3D" python export_object_trajectory.py --layout "$COMBINED/layout_camera_frame.json" \
    --timestamps-ns "$OUT/timestamps_ns.npy" --object "$OBJECT" --mesh "$MESH" \
    --fixed-mesh "$FIXED_MESH" --output "$TRAJECTORY" "${CAMERA[@]}"
echo "Object-only pipeline complete."
