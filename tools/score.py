"""Score a results.jsonl against an every-frame reference (brief, section 5).

python tools/score.py --ref ref/ --pred results.jsonl [--per-video]

Only videos present in the reference are scored. Gates:
  object F1 >= 0.95, people exact >= 85%, people within 1 >= 95%.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vidx.aggregate import per_second  # noqa: E402


def load_ref(ref_dir):
    ref = {}
    for p in sorted(glob.glob(os.path.join(ref_dir, "*.npz"))):
        z = np.load(p)
        vid = os.path.basename(p)[:-4].replace("__", ":")
        last = int(z["last_second"])
        if last < 0:
            continue
        ref[vid] = per_second(z["seconds"], z["counts"], last)
    return ref


def load_pred(path, vids):
    pred = {v: {} for v in vids}
    errors = {}
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            v = r.get("video")
            if v not in pred:
                continue
            if "error" in r:
                errors[v] = r["error"]
                continue
            pred[v][int(r["second"])] = (sorted(r["objects"]), int(r["people"]))
    return pred, errors


def score(ref, pred):
    tp = fp = fn = 0
    exact = within1 = total = 0
    per_video = {}
    for v, secs in ref.items():
        p = pred.get(v, {})
        vtp = vfp = vfn = vex = vw1 = 0
        for s, (objs, people) in enumerate(secs):
            po, pp = p.get(s, ([], None))
            ro, so = set(objs), set(po)
            vtp += len(ro & so)
            vfp += len(so - ro)
            vfn += len(ro - so)
            if pp is not None and pp == people:
                vex += 1
            if pp is not None and abs(pp - people) <= 1:
                vw1 += 1
        extra = [s for s in p if s >= len(secs)]
        for s in extra:
            vfp += len(p[s][0])
        tp, fp, fn = tp + vtp, fp + vfp, fn + vfn
        exact, within1, total = exact + vex, within1 + vw1, total + len(secs)
        per_video[v] = dict(tp=vtp, fp=vfp, fn=vfn, secs=len(secs), exact=vex,
                            within1=vw1, extra_secs=len(extra))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return dict(f1=f1, exact=exact / max(1, total), within1=within1 / max(1, total),
                seconds=total, videos=len(ref), tp=tp, fp=fp, fn=fn), per_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--per-video", action="store_true")
    args = ap.parse_args()
    ref = load_ref(args.ref)
    pred, errors = load_pred(args.pred, ref.keys())
    s, pv = score(ref, pred)
    ok = s["f1"] >= 0.95 and s["exact"] >= 0.85 and s["within1"] >= 0.95
    print(f"videos={s['videos']} seconds={s['seconds']} "
          f"F1={s['f1']:.4f} exact={s['exact']:.4f} within1={s['within1']:.4f} "
          f"(tp={s['tp']} fp={s['fp']} fn={s['fn']}) errors={len(errors)} "
          f"{'PASS' if ok else 'FAIL'}")
    if args.per_video:
        for v, d in pv.items():
            print(v, d, errors.get(v, ""))


if __name__ == "__main__":
    main()
