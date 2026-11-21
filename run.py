"""Per-second object presence and people counts for FineVideo Parquet files.

Usage: python run.py --input <folder of .parquet files> --output results.jsonl

Default mode is the fast pipeline (parallel decode workers + TensorRT FP16 on
the GPU). `--mode simple` runs the first, single-process version (Ultralytics
predict) for comparison.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vidx.aggregate import per_second  # noqa: E402
from vidx.parquet_io import iter_videos, list_parquet  # noqa: E402
from vidx.video import VideoReader  # noqa: E402


def effective_cpus():
    """CPUs we may actually use: affinity, limited by a cgroup CPU quota."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n)


# ---------------------------------------------------------------- simple mode
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


def run_simple(args):
    from vidx.detect_ultra import UltralyticsDetector

    detector = UltralyticsDetector(args.weights, batch=32)
    t0 = time.time()
    n_videos = 0
    with open(args.output, "w") as out:
        for path in list_parquet(args.input):
            for vid, data, err in iter_videos(path):
                n_videos += 1
                if err:
                    records = [{"video": vid, "error": err}]
                else:
                    records = process_video(vid, data, detector, args.per_second, 32)
                for r in records:
                    out.write(json.dumps(r) + "\n")
                out.flush()
            print(f"[{time.time() - t0:8.1f}s] {os.path.basename(path)} done, "
                  f"{n_videos} videos", file=sys.stderr, flush=True)
    print(f"total {time.time() - t0:.1f}s for {n_videos} videos", file=sys.stderr)


# ------------------------------------------------------------------ fast mode
def run_fast(args):
    from vidx.engine import load_model
    from vidx.pipeline import log, run

    n_workers = args.workers or max(1, effective_cpus() - 1)
    log(f"workers: {n_workers}, frames/s sampled: {args.per_second}, "
        f"sequential: {not args.no_sequential}, skip non-ref frames: {args.skip_nonref}")

    def get_model():
        return load_model(args.weights, args.engine, log=log)

    run(args.input, args.output, model_factory=get_model, n_workers=n_workers,
        per_second_k=args.per_second, skip_nonref=args.skip_nonref,
        sequential=not args.no_sequential, max_batch=args.batch, debug=args.debug)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl path")
    ap.add_argument("--mode", choices=["fast", "simple"], default="fast")
    ap.add_argument("--weights", default=os.path.join(HERE, "yolo11l.pt"))
    ap.add_argument("--engine", default=os.path.join(HERE, "yolo11l.engine"))
    ap.add_argument("--per-second", type=int, default=3,
                    help="frames analysed per second (0 = every frame)")
    ap.add_argument("--skip-nonref", action="store_true",
                    help="do not decode non-reference (B) frames")
    ap.add_argument("--no-sequential", action="store_true",
                    help="analyse all sampled frames (no sequential sampling)")
    ap.add_argument("--workers", type=int, default=0, help="decode processes (0 = auto)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.mode == "simple":
        run_simple(args)
    else:
        run_fast(args)


if __name__ == "__main__":
    main()
