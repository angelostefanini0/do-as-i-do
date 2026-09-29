#!/usr/bin/env python3
"""Track metric MPS chair landmarks with BootsTAPIR; keep original RGB indices.

Run in the existing tapnet environment. This only produces 2D correspondences;
the calibrated PnP stage separately validates and estimates the rigid motion.
"""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from tapnet.torch import tapir_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True, type=Path)
    parser.add_argument('--landmarks', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--query-chunk-size', type=int, default=16)
    args = parser.parse_args()
    landmarks = np.load(args.landmarks)
    uv = landmarks['reference_uv'].astype(np.float32)
    reference = int(landmarks['reference_frame'])
    paths = sorted((args.sequence / 'all_frames').glob('*.png'))
    print(f'Loading {len(paths)} RGB frames and {len(uv)} metric landmarks', flush=True)
    with Image.open(paths[0]) as im:
        width, height = im.size
    video = np.stack([np.asarray(Image.open(p).convert('RGB').resize((args.resize, args.resize), Image.Resampling.BILINEAR)) for p in paths])
    queries = np.column_stack([np.full(len(uv), reference), uv[:, 1] * args.resize / height, uv[:, 0] * args.resize / width]).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to(device).eval()
    start = time.monotonic()
    print(f'Tracking on {device}; query frame {reference}; resolution {args.resize}', flush=True)
    with torch.inference_mode():
        frames = torch.from_numpy(video).to(device).float() / 255. * 2. - 1.
        result = model(frames[None], torch.from_numpy(queries).to(device)[None], query_chunk_size=args.query_chunk_size)
        tracks = result['tracks'][0].cpu().numpy()
        confidence = ((1. - result['occlusion'][0].sigmoid()) * (1. - result['expected_dist'][0].sigmoid())).cpu().numpy()
    tracks *= np.array([width / args.resize, height / args.resize], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tracks_uv=tracks, confidence=confidence, reference_frame=reference, source_uids=landmarks['uids'], timestamps_ns=np.load(args.sequence / 'timestamps_ns.npy'), image_size=np.array([width, height]), runtime_s=time.monotonic() - start)
    print(f'Saved {tracks.shape}; median visible points per frame: {np.median((confidence>.5).sum(axis=0)):.1f}; elapsed {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
