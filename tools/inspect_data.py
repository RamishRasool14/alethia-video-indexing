"""Survey the videos: codec, size, fps, duration, rotation, B-frames, row groups.

python tools/inspect_data.py --input finevideo/data --out survey.csv
"""
import argparse
import csv
import io
import os
import sys
from multiprocessing import Pool

import av
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vidx.parquet_io import iter_videos, list_parquet  # noqa: E402


def probe(path):
    rows = []
    pf = pq.ParquetFile(path)
    groups = pf.num_row_groups
    for vid, data, err in iter_videos(path):
        r = dict(video=vid, row_groups=groups, bytes=len(data or b""), error=err or "")
        if data:
            try:
                with av.open(io.BytesIO(data)) as c:
                    s = c.streams.video[0]
                    cc = s.codec_context
                    r.update(codec=cc.name, width=cc.width, height=cc.height,
                             pix_fmt=cc.format.name if cc.format else "",
                             avg_rate=float(s.average_rate or 0),
                             frames=s.frames,
                             duration=float(s.duration * s.time_base) if s.duration else
                             (c.duration / 1e6 if c.duration else 0),
                             has_b=getattr(cc, "has_b_frames", ""),
                             rotate=(s.metadata or {}).get("rotate", ""))
                    f = next(c.decode(s))
                    r["frame_rotation"] = getattr(f, "rotation", "")
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {e}"[:100]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    with Pool(args.workers) as p:
        allrows = [r for rows in p.map(probe, list_parquet(args.input)) for r in rows]
    keys = sorted({k for r in allrows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(allrows)
    print(f"{len(allrows)} videos -> {args.out}")


if __name__ == "__main__":
    main()
