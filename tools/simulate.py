"""Simulate frame sampling offline from the every-frame reference.

For each K, pick frames exactly as VideoReader(per_second=K) would (first
frame in each 1/K-second slot), aggregate with the same rules, and score
against the every-frame result. This isolates the error caused by sampling
from the error caused by numeric differences (FP16, TensorRT, resizing).

python tools/simulate.py --ref ref/ --k 1 2 3 4 5 6 8 10
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.score import score  # noqa: E402
from vidx.aggregate import per_second  # noqa: E402


def pick(times, seconds, k):
    keep, last = [], None
    for i, (t, s) in enumerate(zip(times, seconds)):
        slot = (int(s), math.floor((t - s) * k + 1e-9))
        if slot != last:
            keep.append(i)
            last = slot
    return np.array(keep, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    args = ap.parse_args()
    data = {}
    for p in sorted(glob.glob(os.path.join(args.ref, "*.npz"))):
        z = np.load(p)
        if int(z["last_second"]) < 0:
            continue
        data[os.path.basename(p)[:-4]] = (z["times"], z["seconds"], z["counts"],
                                          int(z["last_second"]))
    ref = {v: per_second(s, c, last) for v, (t, s, c, last) in data.items()}
    n_frames = sum(len(t) for t, *_ in data.values())
    for k in args.k:
        pred, used = {}, 0
        for v, (t, s, c, last) in data.items():
            idx = pick(t, s, k)
            used += len(idx)
            pred[v] = dict(enumerate(per_second(s[idx], c[idx], last)))
        r, _ = score(ref, pred)
        ok = r["f1"] >= 0.95 and r["exact"] >= 0.85 and r["within1"] >= 0.95
        print(f"K={k:2d} frames={used / n_frames:6.1%} F1={r['f1']:.4f} "
              f"exact={r['exact']:.4f} within1={r['within1']:.4f} "
              f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
