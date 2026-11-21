"""Compare backends on identical frames and time the GPU path.

- per-frame count agreement: Ultralytics predict (FP32, the reference path)
  vs our letterbox + TensorRT FP16 + GPU NMS, and vs PyTorch FP16.
- throughput of the TensorRT engine at several batch sizes.

python tools/bench_gpu.py --input finevideo/data --frames 1024
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vidx.detect_ultra import UltralyticsDetector  # noqa: E402
from vidx.engine import TorchModel, TRTModel  # noqa: E402
from vidx.letterbox import PAD, geometry, letterbox_into  # noqa: E402
from vidx.parquet_io import iter_videos, list_parquet  # noqa: E402
from vidx.postprocess import count_classes  # noqa: E402
from vidx.video import VideoReader  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_input(batch_u8):
    x = torch.from_numpy(batch_u8).cuda()
    return x.permute(0, 3, 1, 2).flip(1).float().div_(255)


def ours(model, frames, bs=64):
    out = []
    for i in range(0, len(frames), bs):
        chunk = frames[i:i + bs]
        h, w = chunk[0].shape[:2]
        g = geometry(h, w)
        buf = np.full((len(chunk), g[0], g[1], 3), PAD, np.uint8)
        for j, f in enumerate(chunk):
            letterbox_into(f, buf[j], g)
        out.append(count_classes(model(to_input(buf))).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--file", type=int, default=3)
    args = ap.parse_args()
    frames = []
    for vid, data, err in iter_videos(list_parquet(args.input)[args.file]):
        if not data:
            continue
        for _, img in VideoReader(data, per_second=3):
            frames.append(img)
        if len(frames) >= args.frames:
            break
    frames = frames[: args.frames]
    shapes = {f.shape for f in frames}
    print(f"{len(frames)} frames, shapes {shapes}")
    # group by shape so every batch has one shape (as in Ultralytics' rect mode)
    by_shape = {}
    for f in frames:
        by_shape.setdefault(f.shape, []).append(f)
    frames = [f for fs in by_shape.values() for f in fs]

    ref = np.concatenate([UltralyticsDetector(os.path.join(HERE, "yolo11l.pt")).counts(fs)
                          for fs in by_shape.values()])
    trt_model = TRTModel(os.path.join(HERE, "yolo11l.engine"))
    a = np.concatenate([ours(trt_model, fs) for fs in by_shape.values()])
    tm = TorchModel(os.path.join(HERE, "yolo11l.pt"), half=True)
    b = np.concatenate([ours(tm, fs) for fs in by_shape.values()])
    tm32 = TorchModel(os.path.join(HERE, "yolo11l.pt"), half=False)
    c = np.concatenate([ours(tm32, fs) for fs in by_shape.values()])
    for name, x in [("TRT fp16", a), ("torch fp16", b), ("torch fp32 (our pre/post)", c)]:
        same = (x == ref).all(1).mean()
        print(f"{name:28s}: frames identical to Ultralytics {same:.4f}, "
              f"per-class mismatches {(x != ref).sum(0).tolist()}, "
              f"totals {x.sum(0).tolist()} vs {ref.sum(0).tolist()}")

    # throughput, synthetic input at 384x640
    for bs in (16, 32, 64):
        x = torch.rand(bs, 3, 384, 640, device="cuda")
        for _ in range(3):
            count_classes(trt_model(x))
        torch.cuda.synchronize()
        n = 30
        t = time.time()
        for _ in range(n):
            count_classes(trt_model(x)).cpu()
        torch.cuda.synchronize()
        print(f"TRT bs={bs}: {bs * n / (time.time() - t):.0f} fps (384x640, incl. NMS)")
    x = torch.rand(64, 3, 384, 640, device="cuda")
    tm(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        count_classes(tm(x)).cpu()
    print(f"torch fp16 bs=64: {640 / (time.time() - t):.0f} fps")


if __name__ == "__main__":
    main()
