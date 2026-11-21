"""Experimental: build an INT8 (+FP16) TensorRT engine calibrated on frames
sampled from the dataset, for accuracy/speed comparison with the FP16 engine.

python tools/build_int8.py --input finevideo/data --out yolo11l_int8.engine --frames 1024
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vidx.engine import build_engine  # noqa: E402
from vidx.letterbox import PAD, geometry, letterbox_into  # noqa: E402
from vidx.parquet_io import iter_videos, list_parquet  # noqa: E402
from vidx.video import VideoReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=1024)
    args = ap.parse_args()
    files = list_parquet(args.input)
    frames = []
    per_video = 8
    # calibration frames from files 40-49 (the reference sample is mostly elsewhere)
    for path in files[40:]:
        for vid, data, err in iter_videos(path):
            if not data:
                continue
            got = 0
            for sec, img in VideoReader(data, per_second=1):
                if img.shape[:2] != (360, 640) or sec % 15:
                    continue
                g = geometry(360, 640)
                buf = np.full((g[0], g[1], 3), PAD, np.uint8)
                letterbox_into(img, buf, g)
                frames.append(buf)
                got += 1
                if got == per_video:
                    break
            if len(frames) >= args.frames:
                break
        if len(frames) >= args.frames:
            break
    frames = np.stack(frames[: args.frames])
    print("calibration frames", frames.shape)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_engine(os.path.join(here, "yolo11l.onnx"), args.out, int8_frames=frames,
                 int8_cache=args.out + ".calib")
    print("built", args.out)


if __name__ == "__main__":
    main()
