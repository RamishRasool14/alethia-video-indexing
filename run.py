"""Per-second object presence and people counts for FineVideo Parquet files.

Usage: python run.py --input <folder of .parquet files> --output results.jsonl
"""
import argparse
import json
import os
import sys
import time

from vidx.aggregate import per_second
from vidx.parquet_io import iter_videos, list_parquet
from vidx.video import VideoReader

HERE = os.path.dirname(os.path.abspath(__file__))


def process_video(vid, data, detector, per_sec, batch):
    """Return the output records for one video (never raises)."""
    seconds, counts, buf, buf_sec = [], [], [], []
    reader = VideoReader(data, per_second=per_sec)
    error = None

    def flush():
        if buf:
            counts.extend(detector.counts(buf))
            seconds.extend(buf_sec)
            buf.clear()
            buf_sec.clear()

    try:
        for sec, img in reader:
            buf.append(img)
            buf_sec.append(sec)
            if len(buf) >= batch:
                flush()
        flush()
    except Exception as e:
        error = f"could not decode video: {type(e).__name__}"
        try:
            flush()
        except Exception:
            buf.clear()
            buf_sec.clear()
    records = []
    if reader.last_second >= 0:
        for s, (objects, people) in enumerate(
            per_second(seconds[: len(counts)], counts, reader.last_second)
        ):
            records.append({"video": vid, "second": s, "objects": objects, "people": people})
    elif error is None:
        error = "no frames decoded"
    if error:
        records.append({"video": vid, "error": error})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl path")
    ap.add_argument("--weights", default=os.path.join(HERE, "yolo11l.pt"))
    ap.add_argument("--per-second", type=int, default=0,
                    help="frames analysed per second (0 = every frame)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from vidx.detect_ultra import UltralyticsDetector

    detector = UltralyticsDetector(args.weights, batch=args.batch)
    t0 = time.time()
    n_videos = 0
    with open(args.output, "w") as out:
        for path in list_parquet(args.input):
            for vid, data, err in iter_videos(path):
                n_videos += 1
                if err:
                    records = [{"video": vid, "error": err}]
                else:
                    records = process_video(vid, data, detector, args.per_second, args.batch)
                for r in records:
                    out.write(json.dumps(r) + "\n")
                out.flush()
            print(f"[{time.time() - t0:8.1f}s] {os.path.basename(path)} done, "
                  f"{n_videos} videos", file=sys.stderr, flush=True)
    print(f"total {time.time() - t0:.1f}s for {n_videos} videos", file=sys.stderr)


if __name__ == "__main__":
    main()
