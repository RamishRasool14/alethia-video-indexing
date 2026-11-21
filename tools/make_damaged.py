"""Build a Parquet file of damaged and unusual videos for robustness tests.

python tools/make_damaged.py --src finevideo/data/train-00000-of-01357.parquet --out tests_pq/

Rows (needs ffmpeg on PATH for the re-encoded cases):
  0 good video (unchanged)
  1 truncated to 50% of its bytes
  2 truncated to 5% of its bytes
  3 random bytes overwritten in the middle (bit rot)
  4 header only (first 4 KB)
  5 random garbage
  6 empty bytes
  7 null
  8 audio-only mp4 (no video stream)
  9 rotated 90 degrees (display matrix)
 10 rotated 270 degrees
 11 variable frame rate (timestamps jittered)
 12 MPEG-4 Part 2 codec instead of H.264
 13 HEVC (H.265)
 14 VP9 in mp4
 15 odd size 333x197
 16 single frame
 17 1080p (larger than training data)
Also writes a second file that is not a valid Parquet file.
"""
import argparse
import os
import random
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def ffmpeg(args, src_bytes, suffix=".mp4"):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.mp4")
        dst = os.path.join(d, "out" + suffix)
        open(src, "wb").write(src_bytes)
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", src] + args + [dst]
        try:
            subprocess.run(cmd, check=True, timeout=300)
            return open(dst, "rb").read()
        except Exception as e:
            print("ffmpeg failed:", " ".join(args), e)
            return b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rnd = random.Random(0)
    col = pq.ParquetFile(args.src).read_row_group(0, columns=["mp4"]).column("mp4")
    # a short video keeps the re-encodes quick
    vids = sorted((col[i].as_py() for i in range(len(col))), key=len)
    good = vids[len(vids) // 4]
    short = ffmpeg(["-t", "20", "-c", "copy"], good)

    def corrupt(b):
        b = bytearray(b)
        for _ in range(200):
            i = rnd.randrange(len(b) // 4, len(b))
            b[i] = rnd.randrange(256)
        return bytes(b)

    vfr = ffmpeg(["-t", "20", "-vf", "setpts='PTS+0.02*sin(N)/TB'", "-vsync", "vfr",
                  "-c:v", "libx264", "-an"], good)
    rows = [
        good,
        good[: len(good) // 2],
        good[: len(good) // 20],
        corrupt(good),
        good[:4096],
        bytes(rnd.randrange(256) for _ in range(100000)),
        b"",
        None,
        ffmpeg(["-t", "10", "-vn", "-c:a", "aac"], good),
        ffmpeg(["-c", "copy", "-metadata:s:v:0", "rotate=90"], short),
        ffmpeg(["-c", "copy", "-metadata:s:v:0", "rotate=270"], short),
        vfr,
        ffmpeg(["-c:v", "mpeg4", "-q:v", "5", "-an"], short),
        ffmpeg(["-c:v", "libx265", "-an"], short),
        ffmpeg(["-c:v", "libvpx-vp9", "-b:v", "500k", "-deadline", "realtime", "-an"], short),
        ffmpeg(["-vf", "scale=333:197", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-an"], short),
        ffmpeg(["-frames:v", "1", "-c:v", "libx264", "-an"], short),
        ffmpeg(["-vf", "scale=1920:1080", "-c:v", "libx264", "-preset", "ultrafast", "-an"],
               short),
    ]
    table = pa.table({"mp4": pa.array(rows, pa.binary()),
                      "json": pa.array(["{}"] * len(rows))})
    pq.write_table(table, os.path.join(args.out, "damaged-00000.parquet"))
    with open(os.path.join(args.out, "damaged-00001.parquet"), "wb") as f:
        f.write(b"PAR1" + bytes(rnd.randrange(256) for _ in range(5000)))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
