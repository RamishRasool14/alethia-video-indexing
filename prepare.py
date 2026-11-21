"""One-time preparation (run with internet, before the timed run).

1. Downloads the official yolo11l.pt weights next to run.py.
2. Exports them to ONNX and builds a TensorRT FP16 engine for this GPU
   (yolo11l.engine). If this step fails, run.py falls back to PyTorch FP16.
"""
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"


def main():
    weights = os.path.join(HERE, "yolo11l.pt")
    if not os.path.exists(weights):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, weights + ".part")
        os.replace(weights + ".part", weights)
    print(f"weights: {weights} ({os.path.getsize(weights) / 1e6:.1f} MB)")

    sys.path.insert(0, HERE)
    from vidx.engine import build_engine, export_onnx

    onnx_path = os.path.join(HERE, "yolo11l.onnx")
    engine_path = os.path.join(HERE, "yolo11l.engine")
    t = time.time()
    if not os.path.exists(onnx_path):
        export_onnx(weights, onnx_path)
    print(f"onnx: {onnx_path}")
    build_engine(onnx_path, engine_path + ".part")
    os.replace(engine_path + ".part", engine_path)
    print(f"engine: {engine_path} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
