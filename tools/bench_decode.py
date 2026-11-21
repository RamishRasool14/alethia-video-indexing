"""Decode throughput: threads vs processes, skip_frame, frame conversion cost.

python tools/bench_decode.py --input finevideo/data --n 12 --workers 1 4 8 12 16
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import av

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vidx.parquet_io import iter_videos, list_parquet  # noqa: E402

VIDEOS = []


def decode(i, skip="DEFAULT", convert_every=0, threads=1):
    data = VIDEOS[i]
    n = 0
    with av.open(io.BytesIO(data)) as c:
        s = c.streams.video[0]
        s.codec_context.skip_frame = skip
        if threads > 1:
            s.thread_type = "AUTO"
            s.thread_count = threads
        for f in c.decode(s):
            if convert_every and n % convert_every == 0:
                f.to_ndarray(format="bgr24")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 12, 16])
    args = ap.parse_args()
    for vid, data, err in iter_videos(list_parquet(args.input)[0]):
        if data:
            VIDEOS.append(data)
        if len(VIDEOS) >= args.n:
            break
    idx = list(range(len(VIDEOS)))

    def run(pool_cls, w, **kw):
        t = time.time()
        with pool_cls(w) as p:
            n = sum(p.map(_call, [(i, kw) for i in idx]))
        return n / (time.time() - t)

    n1 = decode(0)
    t = time.time(); decode(0); single = n1 / (time.time() - t)
    t = time.time(); decode(0, skip="NONREF"); nonref = n1 / (time.time() - t)
    t = time.time(); decode(0, convert_every=1); conv = n1 / (time.time() - t)
    t = time.time(); decode(0, convert_every=6); conv6 = n1 / (time.time() - t)
    print(f"1 video, 1 thread: decode {single:.0f} fps, NONREF {nonref:.0f} fps "
          f"(as source frames), +bgr every frame {conv:.0f} fps, +bgr every 6th {conv6:.0f} fps")
    for w in args.workers:
        print(f"workers={w:2d}: threads {run(ThreadPoolExecutor, w):6.0f} fps | "
              f"processes {run(ProcessPoolExecutor, w):6.0f} fps | "
              f"processes NONREF {run(ProcessPoolExecutor, w, skip='NONREF'):6.0f} fps",
              flush=True)


def _call(a):
    i, kw = a
    return decode(i, **kw)


if __name__ == "__main__":
    main()
