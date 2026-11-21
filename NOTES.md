# Engineering log

Plain record of what was run, what changed, and the measured numbers.
Times are wall-clock on the RunPod pod described in HARDWARE.md unless noted.
"Gates" = brief section 5: object F1 >= 0.95, people exact >= 85%, within 1 >= 95%.

## Pod

- RunPod Secure cloud, RTX 4090 24 GB, driver 595.112.07, AMD EPYC 7452
  (container CPU quota 13.6 cores, 16 vCPU allocated), 1 TB host RAM visible
  (125 GB allocated), Ubuntu 22.04.5, Python 3.11.10. Created 21 Nov 12:54 PKT.
- Dataset download (`hf download ... train-000[0-4]*`): 50 files, 24 GB, 31 s.

## Data survey (tools/inspect_data.py, all 50 files)

- 1,650 videos, 128.0 hours, 12,981,615 frames. 0 videos failed to open.
- Codec: H.264 for all 1,650; pixel format yuv420p for all.
- Sizes: 640x360 = 1,284; 360x640 (portrait) = 141; 480x360 = 49; 270x480 = 31;
  202x360 = 28; others < 10 each. Largest dimension is 640 everywhere.
- Frame rates: 29.97 = 575, 30 = 505, 25 = 265, 23.98 = 177, 24 = 71, plus a few
  at 7-15 fps.
- Rotation tags: none in these 50 files.
- B-frames present in 1,376 of 1,650 videos.
- Each Parquet file is a single row group (~500 MB).
- Duration per video: 19 s to 658 s, median 242 s.

## Checks on synthetic inputs

- Rotation: test video tagged 90/180/270 degrees. Our PyAV frames after rotation
  are pixel-identical (mean abs diff 0.0) to OpenCV's auto-rotated frames and
  to ffmpeg CLI output.
- Damaged inputs (first version): unreadable Parquet file, truncated mp4, random
  bytes, empty bytes, null row -> one error line each, run continues.

## Reference (tools/reference.py)

- 59 videos sampled (44 random + portrait, odd sizes, low fps, no-B-frame cases),
  4.15 hours, 412,151 frames, 14,958 seconds.
- Plain Ultralytics predict, FP32, every frame. 3 processes sharing the GPU,
  ~300 frames/s total, ~25 min.

## Sampling simulation (tools/simulate.py, reference per-frame counts)

Frames analysed per second (K), taking the first frame in each 1/K-second slot,
aggregated with the brief's rules, scored against all frames:

| K | frames used | F1 | exact | within 1 |
|---|---|---|---|---|
| 1 | 3.6% | 0.9528 | 0.8954 | 0.9811 |
| 2 | 7.3% | 0.9654 | 0.9241 | 0.9952 |
| 3 | 10.9% | 0.9826 | 0.9502 | 0.9968 |
| 4 | 14.5% | 0.9828 | 0.9551 | 0.9989 |
| 5 | 18.1% | 0.9891 | 0.9673 | 0.9985 |
| 6 | 21.7% | 0.9886 | 0.9690 | 0.9994 |
| 8 | 28.9% | 0.9919 | 0.9758 | 0.9997 |
| 10 | 36.1% | 0.9936 | 0.9801 | 1.0000 |
| 15 | 53.8% | 0.9967 | 0.9899 | 0.9998 |

## Micro-benchmarks

- Decode (PyAV, 360p H.264, one core): ~2,100 frames/s; with skip_frame=NONREF
  (B-frames skipped) a video is processed ~1.8x faster. +bgr24 conversion of
  every frame: ~1,600 frames/s.
- Decode in parallel (24 videos): 4 processes 6,856 fps; 8 processes 11,466 fps;
  12 processes 14,611 fps. Threads: 8 -> 10,493 fps, 12 -> 9,468 fps.
  (Measured while the reference used ~3 cores.)
- TensorRT 10.9 FP16 engine, dynamic shape optimised for 384x640, incl. GPU NMS:
  bs16 1,817 fps, bs32 1,989 fps, bs64 1,996 fps. PyTorch FP16 bs64: 921 fps.
  Engine build took 548 s (GPU shared with reference at the time).
- Same 1,024 frames, per-frame counts identical to Ultralytics FP32 predict:
  TensorRT FP16 99.41%, PyTorch FP16 99.80%, PyTorch FP32 with our own
  letterbox/NMS 99.90%.

## Runs

| Start (PKT) | Version | Data | Wall time | F1 | Exact | Within 1 | Notes |
|---|---|---|---|---|---|---|---|
| 21 Nov 13:33 | v1-simple (tag), K=3, 1 process, Ultralytics FP32 bs32 | 50 files | stopped after 340 s | | | | Processed 12,838 video-seconds in 340 s (~37.8 video-s per s), i.e. ~3.4 h projected for 461,757 s. Stopped to free the GPU for development; to be re-run in full, undisturbed. |
| 21 Nov 14:20 | fast v2: 13 decode processes + shared memory + TensorRT FP16 bs64 + GPU NMS, K=3 | 50 files | 940.5 s (internal timer; excludes ~20 s start-up) | 0.9824 | 0.9495 | 0.9967 | 1,383,597 frames analysed, 1,474 frames/s. CPU-bound: 13 workers decoding every frame. GPU ~80% busy. 0 error lines. |

| 21 Nov 14:45 | v3: + 2 GPU inference threads (own TensorRT context + CUDA stream each), pinned shared memory, sequential sampling, --skip-nonref, venv on local disk | 50 files | 589.0 s wall | 0.9828 | 0.9516 | 0.9969 | 1,008,201 frames analysed (deferred middle frames run: 86,588 of 460,639 = 18.8%). 1,735 frames/s; GPU-bound on this pod (3 batches in flight, frames pending). 0 error lines, 461,757 lines. |

| 21 Nov 16:28 | final (commit a40a387): v3 + FP16-input engine built as in prepare.py | 50 files | 620.3 s wall | 0.9827 | 0.9518 | 0.9970 | 1,008,170 frames analysed (deferred run 86,557/460,639). 461,757 lines, 0 errors. Slower than the 589.0 s run despite the faster engine: run-to-run variation on this shared host. -> results.jsonl.gz |
| 21 Nov 16:39 | v1-simple (tag), full undisturbed timed run | 50 files | running | | | | |

## Sequential sampling (K=3)

- Idea: analyse slots 0 and 2 of each second first; analyse slot 1 only if those
  two frames disagree on any of the 5 classes or on the people count. If they
  agree, the majority/median over 3 frames is already fixed, so the output is
  identical to analysing all 3.
- Simulation on the reference: K=3 needs 73% of the frames (K=5: 69%, K=7: 69%).
- 3 files (99 videos, 30,447 output lines): output identical line-for-line to the
  non-sequential run (30,447/30,447). Frames analysed 66,950 vs 91,234 (73%).
  Only 6,124 of 30,408 middle frames (20%) were needed. 63.4 s vs 76.3 s
  (still CPU/decode-bound on this pod).
- Bug found while testing: a video could be completed and removed inside the batch
  loop, then looked up again -> KeyError; on that error the process also hung at
  exit (multiprocessing queue feeder thread). Fixed both.

## Start-up cost

- The first venv lived on the RunPod network volume (MooseFS): `import torch`
  took ~7.9 s there. CUDA init ~6.7 s including the import; TensorRT engine
  load 2.1 s. Wall time of a 3-file run was ~20 s more than the internal timer.

## Dependencies

- The working venv had sympy 1.14.0 although torch 2.5.1 pins sympy==1.13.1
  (pulled in by later installs: onnxslim 0.1.99 / Ultralytics auto-install of
  onnxruntime). A requirements file frozen from that venv would not install.
  Resolved with `uv pip compile` for Python 3.11: onnxslim 0.1.98, sympy 1.13.1.

## GPU loop (3-file subset, 99 videos)

- Before: one main thread did message handling, host->GPU copy from ordinary
  (pageable) shared memory, TensorRT, NMS and the read-back in series.
  Workers often sat idle waiting for free chunks (top: several at 0% CPU).
  GPU-bound test (15 frames/s, no sequential): 455,956 frames in 310.1 s =
  1,470 frames/s.
- After: 2 inference threads with their own TensorRT execution contexts and
  CUDA streams; shared-memory blocks page-locked with cudaHostRegister. Same
  GPU-bound test: 261.6 s = 1,743 frames/s (peaks 1,807). TensorRT alone
  (synthetic input): ~2,000 frames/s. Output identical.
- With the venv on local disk: torch import 1.81 s, CUDA init 0.27 s,
  torchvision 1.16 s, tensorrt 0.05 s, av/cv2/pyarrow 0.29 s, engine load 0.31 s.
  `model ready` is logged ~3.4-4.3 s after start.

## Decode cost (one core, this pod's EPYC 7452)

- ~1,700 frames/s (~590 us CPU per frame) for decode alone; exact Fraction
  timestamp maths adds nothing measurable; + bgr24 conversion of the 3 sampled
  frames/s: ~670 us/frame.
- skip_frame=NONREF on 12 videos: 34.26 s -> 22.08 s (81,109 -> 46,608 frames
  decoded). NONKEY/NONINTRA: 896 frames only (too sparse).
- In the pipeline on the 3-file subset: default 59.3 s, --skip-nonref 55.1 s.

## Robustness (tools/make_damaged.py, plus test hooks)

18-row damaged Parquet + one invalid Parquet file. Fast and simple modes give the
same structure for every row:
- truncated 50% -> 27 s of lines + error; truncated 5% -> 9 s + error;
- random bytes overwritten (bit rot) -> decodes fully (FFmpeg conceals);
- header only -> "no frames decoded"; random bytes, audio-only (no video stream)
  -> one error line; empty/null -> "empty mp4 bytes";
- rotated 90/270, VFR, MPEG-4 Part 2, HEVC, VP9, 333x197 (yuv444p), single frame,
  1080p -> normal lines;
- invalid Parquet file -> one error line (`<file>:0`).
- Simulated decoder crash (SIGSEGV in a worker) and hang (killed after
  --stall-timeout with no output): first attempt hung the whole run, because
  all workers shared one multiprocessing.Queue whose background feeder thread
  can lose the last message (or hold the shared lock) when the process dies.
  Fixed by giving each worker its own pipe with synchronous sends. Now: worker
  replaced, one error line for that video, other 31 videos' 8,550 lines
  identical to a normal run, run finishes in 27.1 s.

## INT8 experiment (rejected)

- TensorRT INT8+FP16 engine, entropy calibration on 1,024 letterboxed 640x360
  frames from files 40-49 (tools/build_int8.py). Build 1,080 s.
- Speed (synthetic 64x3x384x640, incl. NMS): INT8 3,836 frames/s vs FP16 1,982.
- Full 50 files, same pipeline otherwise: 452.4 s wall (FP16: 589.0 s; this pod
  is CPU-bound with INT8). Accuracy vs reference: F1 0.9537, exact 0.9017,
  within1 0.9801 (tp 10,577, fp 141, fn 885). FP16: F1 0.9828, fn 220.
  Passes, but only 0.004 above the F1 threshold; false negatives x4.
- Not used.

## Defaults after these tests

- B-frame skipping made the default (`--no-skip-nonref` turns it off): faster
  and the per-video number of output seconds was identical to full decoding
  for all 1,650 videos.

## Fresh install test (tools/fresh_install_test.sh, 21 Nov ~16:00 PKT)

Clean copy of the committed repo (git archive HEAD), new Python 3.11 venv,
`pip install -r requirements.txt` (pip check: no broken requirements),
`python prepare.py`, `python run.py` on the 3-file subset.
- First attempt failed: the pod's 30 GB container disk ran out during pip
  install (torch wheels unpack in /tmp). Re-run with TMPDIR and the venv on the
  100 GB volume.
- prepare.py: 397 s (ONNX export 11 s, TensorRT engine build 385 s). No
  Ultralytics auto-install during export (onnxruntime already pinned).
- run.py on 3 files: 47.1 s internal, 63.4 s wall (venv on the network volume,
  so imports are slow here). 30,447 lines.
- Versus the development engine: 84 of 30,447 lines differ (0.28%), all small
  people-count or presence flips. Each TensorRT build may pick different FP16
  kernels, so outputs near thresholds can change between builds.

## Engine variants (benchmark: 64x384x640 uint8 -> pre-processing -> TensorRT -> NMS, best of 3)

| Engine | Build time | frames/s |
|---|---|---|
| FP16, FP32 input (fresh build) | 385 s | 1,961-1,965 |
| FP16, FP16 input, default optimisation level | 366 s | 1,997 |
| FP16, FP16 input, builder optimisation level 5 | 1,484 s | 1,998-2,005 |

prepare.py now builds the FP16-input engine at the default level.
