"""Build an every-frame reference for a sample of videos.

Runs plain Ultralytics YOLO11l (FP32, imgsz 640, conf 0.4) on EVERY decoded
frame, as the graders describe, and saves the per-frame counts so sampling
strategies can be simulated offline.

python tools/reference.py --input finevideo/data --videos sample.txt --out ref/
  sample.txt: one "<parquet name>:<row>" per line
"""
import argparse
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vidx.detect_ultra import UltralyticsDetector  # noqa: E402
from vidx.video import VideoReader  # noqa: E402


def read_row(path, row):
    pf = pq.ParquetFile(path)
    start = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if row < start + n:
            col = pf.read_row_group(g, columns=["mp4"]).column("mp4")
            return col[row - start].as_py()
        start += n
    raise IndexError(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="yolo11l.pt")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    det = UltralyticsDetector(args.weights, batch=args.batch, half=False)
    vids = [v.strip() for v in open(args.videos) if v.strip()]
    for vid in vids:
        fname, row = vid.rsplit(":", 1)
        dst = os.path.join(args.out, vid.replace(":", "__") + ".npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        data = read_row(os.path.join(args.input, fname), int(row))
        reader = VideoReader(data, per_second=0)
        secs, times, counts, buf, err = [], [], [], [], ""
        try:
            for sec, img in reader:
                buf.append(img)
                secs.append(sec)
                times.append(reader.last_t)
                if len(buf) == args.batch:
                    counts.append(det.counts(buf))
                    buf = []
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if buf:
            counts.append(det.counts(buf))
        counts = np.concatenate(counts) if counts else np.zeros((0, 5), np.int16)
        np.savez_compressed(
            dst, seconds=np.array(secs[: len(counts)], np.int32), counts=counts,
            times=np.array(times[: len(counts)], np.float64),
            last_second=reader.last_second, n_decoded=reader.n_decoded, error=err,
        )
        print(f"{vid}: {reader.n_decoded} frames, {reader.last_second + 1} s, "
              f"{time.time() - t0:.1f}s {err}", flush=True)


if __name__ == "__main__":
    main()
