# Video indexing at speed

For every second of every FineVideo video in a folder of Parquet files, this
reports which of five objects (person, car, dog, laptop, cell phone) YOLO11l
finds and how many people are visible, as JSON Lines.

## Install

Tested on Ubuntu 22.04, Python 3.11, RTX 4090, NVIDIA driver with CUDA 12
support.

System packages (OpenCV needs `libGL` and `glib` at import time):

```
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

Python packages, inside a fresh virtual environment:

```
pip install -r requirements.txt
```

## Preparation command (run once, with internet; not timed)

```
python prepare.py
```

It downloads the official `yolo11l.pt` next to `run.py`, exports it to ONNX
(dynamic batch and image size) and builds a TensorRT FP16 engine
(`yolo11l.engine`, FP16 input) for the GPU it runs on. The build takes about 5 to 10
minutes. If the engine is missing or cannot be loaded, `run.py` falls back to
PyTorch FP16 automatically (slower, same output format).

## Run

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

No internet access is needed during the run. Progress and a summary go to
stderr.

Options (defaults are the submitted configuration):

| Option | Default | Meaning |
|---|---|---|
| `--per-second K` | 3 | Frames sampled per second (first frame in each 1/K-second slot); 0 = every frame |
| `--no-sequential` | off | Analyse every sampled frame. By default, for odd K, the even slots are analysed first and the odd slots only when those disagree (identical output, fewer frames) |
| `--no-skip-nonref` | off | Decode every frame. By default non-reference (B) frames are not decoded, which makes decoding ~35% cheaper; sampling then uses the decoded frames |
| `--stall-timeout S` | 120 | Kill and replace a decode worker that produces nothing for S seconds (its video gets an error line) |
| `--workers N` | CPUs - 1 | Decode processes |
| `--batch N` | 64 | Frames per GPU batch |
| `--mode simple` | `fast` | The first, single-process version (Ultralytics `predict`), kept for comparison |

## How it works

1. A reader thread reads each Parquet file (the `mp4` column only) and queues
   one task per row. Rows that cannot be read become error lines.
2. Decode worker processes (one per CPU) decode each video with PyAV,
   apply any rotation tag, take timestamps from the first decoded frame,
   keep 3 frames per second (skipping non-reference B-frames, which are
   never decoded), letterbox them exactly like Ultralytics (640,
   stride-32 minimum padding, grey 114) and write them into shared memory.
3. Two GPU threads, each with its own CUDA stream and TensorRT execution
   context, copy batches from pinned shared memory, convert BGR to RGB,
   run YOLO11l and count boxes per class after the same filtering as
   Ultralytics: best class per box, confidence > 0.4, class-aware NMS at
   IoU 0.7.
4. Per second, an object is present if it is found in at least half of the
   analysed frames, and the people count is the median (rounded down at .5).
5. A damaged video gives an error line (after any seconds already decoded);
   a decoder crash only restarts that worker.

## Repository layout

- `run.py`: entry point. `vidx/`: pipeline modules.
- `prepare.py`: weights download and TensorRT engine build.
- `tools/`: every-frame reference builder, scorer for the three thresholds,
  sampling simulator, data survey, benchmarks, damaged-file generator.
- `results.jsonl.gz`: results for the 50 FineVideo files.
- `HARDWARE.md`, `NOTES.md` (engineering log with measurements), `WRITEUP.md`,
  `ai_chats/`.
