#!/usr/bin/env python3
"""Extract native-orientation pinhole RGB frames and timestamp-aligned MPS poses.

Run in an EXISTING environment with projectaria_tools (not the GPU model envs).
No image rotation, no resampling in time, no MPS extrapolation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


def align_mps(path, timestamps_ns, T_device_camera):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len({r["graph_uid"] for r in rows}) != 1:
        raise ValueError("MPS contains multiple world frames; select one graph first")
    times = np.array([int(r["tracking_timestamp_us"]) * 1000 for r in rows], dtype=np.int64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("MPS timestamps must be strictly increasing")
    translations = np.array([[float(r[f"t{a}_world_device"]) for a in "xyz"] for r in rows])
    quats = np.array([[float(r[f"q{a}_world_device"]) for a in "xyzw"] for r in rows])
    valid = (timestamps_ns >= times[0]) & (timestamps_ns <= times[-1])
    # Subtract an integer origin before converting nanoseconds to float seconds.
    seconds = (times - times[0]) * 1e-9
    query = (timestamps_ns[valid] - times[0]) * 1e-9
    T_world_camera = np.full((len(timestamps_ns), 4, 4), np.nan)
    poses = np.tile(np.eye(4), (valid.sum(), 1, 1))
    poses[:, :3, :3] = Slerp(seconds, Rotation.from_quat(quats))(query).as_matrix()
    for axis in range(3):
        poses[:, axis, 3] = np.interp(query, seconds, translations[:, axis])
    # Reject interpolation across missing trajectory intervals or poor quality.
    upper = np.clip(np.searchsorted(times, timestamps_ns[valid], side="right"), 1, len(times)-1)
    quality = np.array([float(r["quality_score"]) for r in rows])
    good = ((times[upper] - times[upper-1]) <= 20_000_000) & (quality[upper] > 0) & (quality[upper-1] > 0)
    indices = np.flatnonzero(valid)
    valid[indices[~good]] = False
    T_world_camera[indices[good]] = poses[good] @ T_device_camera
    return T_world_camera, valid, rows[0]["graph_uid"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mps", type=Path)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--inspect", action="store_true", help="Read metadata only; do not extract frames")
    args = parser.parse_args()
    from projectaria_tools.core import calibration, data_provider

    provider = data_provider.create_vrs_data_provider(str(args.vrs.resolve()))
    stream = provider.get_stream_id_from_label("camera-rgb")
    camera = provider.get_device_calibration().get_camera_calib("camera-rgb")
    count = provider.get_num_data(stream)
    native_width, native_height = camera.get_image_size()
    height = round(args.width * native_height / native_width)
    focal = float(camera.get_focal_lengths()[0]) * args.width / native_width
    pinhole = calibration.get_linear_camera_calibration(args.width, height, focal)
    T_device_camera = camera.get_transform_device_camera().to_matrix()
    time_offset_ns = round(camera.get_time_offset_sec_device_camera() * 1e9)
    if args.inspect:
        print(json.dumps({"vrs": str(args.vrs.resolve()), "num_frames": count,
                          "pinhole_size": [args.width, height], "focal_px": focal,
                          "time_offset_ns": time_offset_ns,
                          "T_device_camera": T_device_camera.tolist()}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = args.output_dir / "all_frames"
    if frames.exists() and any(frames.iterdir()):
        raise FileExistsError(f"Refusing to mix frames with an existing extraction: {frames}")
    frames.mkdir(exist_ok=True)
    timestamps_ns = np.empty(count, dtype=np.int64)
    capture_ns = np.empty(count, dtype=np.int64)
    for index in range(count):
        image, record = provider.get_image_data_by_index(stream, index)
        capture_ns[index] = record.capture_timestamp_ns
        timestamps_ns[index] = capture_ns[index] + time_offset_ns
        rgb = calibration.distort_by_calibration(image.to_numpy_array(), pinhole, camera)
        Image.fromarray(rgb).save(frames / f"{index:06d}.png")
        if index % 50 == 0 or index == count - 1:
            print(f"Extracted {index + 1}/{count}", flush=True)
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("RGB timestamps are not strictly increasing")
    np.save(args.output_dir / "timestamps_ns.npy", timestamps_ns)
    metadata = {"vrs": str(args.vrs.resolve()), "num_frames": count,
                "frame_indices": list(range(count)), "camera_axes": "x-right,y-down,z-forward",
                "image_rotation": "none", "pinhole_size": [args.width, height],
                "focal_px": focal, "principal_point_px": pinhole.get_principal_point().tolist(),
                "timestamp_domain": "Aria device clock, exposure midpoint, nanoseconds",
                "time_offset_ns": time_offset_ns, "T_device_camera": T_device_camera.tolist(),
                "calibration": "factory RGB calibration from VRS"}
    if args.mps:
        world, valid, graph_uid = align_mps(args.mps, timestamps_ns, T_device_camera)
        np.savez_compressed(args.output_dir / "aria_camera_trajectory.npz",
                            timestamps_ns=timestamps_ns, capture_timestamps_ns=capture_ns,
                            T_world_camera=world, valid=valid, graph_uid=graph_uid,
                            translation_units="meters", camera_axes=metadata["camera_axes"])
        metadata.update(mps=str(args.mps.resolve()), mps_valid_frames=int(valid.sum()),
                        mps_invalid_frame_indices=np.flatnonzero(~valid).tolist(), graph_uid=graph_uid)
        print(f"MPS coverage: {valid.sum()}/{count}; missing poses are NaN, never extrapolated")
    (args.output_dir / "aria_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
