"""Per-frame cost inside one decode worker, step by step (single process).

python tools/bench_worker.py --input finevideo/data --n 6
"""
import argparse
import io
import math
import os
import resource
import sys
import time
from fractions import Fraction

import av

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vidx.parquet_io import iter_videos, list_parquet  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def variant(data, mode, threads=None, skip=None):
    n = 0
    with av.open(io.BytesIO(data)) as c:
        s = c.streams.video[0]
        if threads is not None:
            s.codec_context.thread_count = threads
        if skip:
            s.codec_context.skip_frame = skip
        tb = s.time_base
        first = None
        last_slot = None
        if mode == "decode":
            for f in c.decode(s):
                n += 1
        elif mode == "demux":
            for p in c.demux(s):
                for f in p.decode():
                    n += 1
        elif mode in ("fraction", "full"):
            for p in c.demux(s):
                for f in p.decode():
                    n += 1
                    if first is None:
                        first = f.pts
                    t = (f.pts - first) * tb
                    sec = math.floor(t)
                    slot = (sec, math.floor((t - sec) * 3))
                    if slot == last_slot:
                        continue
                    last_slot = slot
                    if mode == "full":
                        f.to_ndarray(format="bgr24")
        elif mode == "int":
            num, den = tb.numerator, tb.denominator
            for p in c.demux(s):
                for f in p.decode():
                    n += 1
                    if first is None:
                        first = f.pts
                    q = (f.pts - first) * num  # time = q / den
                    sec = q // den
                    slot = (sec, (q - sec * den) * 3 // den)
                    if slot == last_slot:
                        continue
                    last_slot = slot
                    f.to_ndarray(format="bgr24")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    vids = []
    for vid, data, err in iter_videos(list_parquet(args.input)[1]):
        if data:
            vids.append(data)
        if len(vids) >= args.n:
            break
    for name, kw in [("decode", {}), ("decode", {"threads": 1}), ("demux", {"threads": 1}),
                     ("fraction", {"threads": 1}), ("full", {"threads": 1}),
                     ("int", {"threads": 1}), ("int", {"threads": 1, "skip": "NONREF"}),
                     ("decode", {"threads": 1, "skip": "NONREF"})]:
        w, c = time.time(), cpu()
        n = sum(variant(d, name, **kw) for d in vids)
        w, c = time.time() - w, cpu() - c
        print(f"{name:9s} {str(kw):38s} {n / w:7.0f} frames/s wall, "
              f"{1e6 * c / n:6.0f} us CPU/frame, cpu/wall {c / w:.2f}")


if __name__ == "__main__":
    main()
