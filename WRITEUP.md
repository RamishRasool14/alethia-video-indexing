# Write-up

## 1. Your timings

First version: 10,961 s (3 h 3 min) for all 50 files on one RTX 4090 pod. It used one process, plain Ultralytics FP32 and 3 frames per second. Final version: 620 s on the same pod, about 17.7x faster, with the same accuracy (F1 0.983, 95.2% exact, 99.7% within 1).

The change that saved the most time was moving decode into 13 parallel worker processes that feed a TensorRT FP16 engine through shared memory. That alone took the 3 hour job to 940 s. Everything after it took 940 s down to about 600 s.

## 2. Your pipeline

1. `prepare.py` exports the model to ONNX and builds a TensorRT FP16 engine (once, about 6 minutes).
2. `run.py` reads each Parquet file and hands videos to 13 decode workers.
3. Each worker decodes with PyAV, skips B-frames, applies rotation tags and keeps 3 frames per second.
4. Frames go into page-locked shared memory for fast copies to the GPU.
5. Two inference threads, each with its own TensorRT context and CUDA stream, run batches of 64 with NMS on the GPU.
6. Sequential sampling: slots 0 and 2 of each second are analysed first. Slot 1 is analysed only if they disagree on a class or the people count; otherwise the majority and median are already fixed. Only 18.8% of middle frames were needed.
7. Per-second results are aggregated and written as JSON lines. A damaged video gives one error line, not a crash.

## 3. Where the time goes

Decode and inference. I measured each stage on its own, then end-to-end.

- Decode: one core does about 1,700 frames/s (590 µs per frame). Skipping B-frames cut decoded frames from 81,109 to 46,608 and time from 34.3 s to 22.1 s on 12 videos. 12 processes reach about 14,600 frames/s.
- Inference: TensorRT FP16 at batch 64, including NMS, does about 2,000 frames/s on synthetic input. PyTorch FP16 does 921.
- End-to-end: the first fast version was CPU-bound at 1,474 frames/s with the GPU about 80% busy. After adding two inference threads and pinned memory it is GPU-bound at about 1,740 frames/s, close to the engine's own limit.
- Start-up: with the venv on the network volume, `import torch` took 7.9 s. On local disk the model is ready in about 4 s.

Timings come from an internal timer plus wall clock, `top` for CPU use, and per-stage micro-benchmarks.

## 4. Checking correctness

I built a reference by running plain Ultralytics FP32 predict on every frame of 59 videos (4.15 h, 412,151 frames), chosen to include portrait, odd sizes, low frame rates and videos without B-frames. Every run is scored against it using the brief's thresholds. The final version gets F1 0.983, 95.2% exact and 99.7% within 1. Sequential sampling was compared line by line with the non-sequential run: 30,447 of 30,447 lines identical. Rotation matches OpenCV and ffmpeg pixel for pixel. Damaged inputs (truncated files, random bytes, worker crashes and hangs) each give one error line, and the run finishes.

## 5. Working with AI

The AI could not create the pod because RunPod returned 401 Unauthorized. It concluded my API key was probably restricted or revoked and told me to create a new one. I knew the key was valid and told it the key was in the shell environment. It checked again and found the real cause: it had been calling RunPod endpoints that are retired, and the v2 API accepted the key. I also questioned its first time estimate of 25–30 hours. It admitted it had overstated it and revised it to 10–14 hours.

## 6. Next steps

1. Make engine builds reproducible. Each TensorRT build picks different FP16 kernels, so 0.28% of output lines changed between builds on a 3-file test. I would pin a timing cache so every build gives the same output.
2. Move decode to the GPU (NVDEC), because the pipeline turns CPU-bound on this pod as soon as inference gets faster.
3. Retry INT8 with better calibration data. It was almost 2x faster, but F1 fell to 0.954.
4. Time several runs and report the median, because near-identical runs on the shared host varied by about 30 s.
