"""One-time preparation (run with internet, before the timed run).

Downloads the official yolo11l.pt weights next to run.py.
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"


def main():
    dst = os.path.join(HERE, "yolo11l.pt")
    if not os.path.exists(dst):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, dst + ".part")
        os.replace(dst + ".part", dst)
    print(f"weights: {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
