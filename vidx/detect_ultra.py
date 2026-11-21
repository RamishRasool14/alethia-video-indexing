"""Detector using the plain Ultralytics predict API (also used for the reference)."""
import numpy as np

from . import CLASS_IDS, CONF, IMGSZ, IOU


class UltralyticsDetector:
    def __init__(self, weights="yolo11l.pt", device=0, half=False, batch=32):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.half = half
        self.batch = batch

    def counts(self, frames):
        """frames: list of HxWx3 uint8 BGR arrays -> int array (n, 5)."""
        out = np.zeros((len(frames), len(CLASS_IDS)), dtype=np.int16)
        for start in range(0, len(frames), self.batch):
            chunk = frames[start:start + self.batch]
            kw = {"half": True} if self.half else {}
            results = self.model.predict(
                chunk,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                classes=CLASS_IDS,
                device=self.device,
                verbose=False,
                **kw,
            )
            for i, r in enumerate(results):
                cls = r.boxes.cls.int().cpu().numpy()
                for j, c in enumerate(CLASS_IDS):
                    out[start + i, j] = int((cls == c).sum())
        return out
