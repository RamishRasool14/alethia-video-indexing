"""Letterbox geometry identical to Ultralytics' LetterBox(auto=True, stride=32).

Ultralytics' predictor uses auto=True (minimum-rectangle padding) when all
images in a batch have the same shape and the model is a PyTorch model, which
is how the reference is produced. A 640x360 frame therefore becomes 640x384
(no resize, 12 px of grey padding top and bottom).
"""
import cv2
import numpy as np

PAD = 114


def geometry(h, w, imgsz=640, stride=32):
    """Return (out_h, out_w, new_h, new_w, top, left) for an h x w frame."""
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = imgsz - new_w, imgsz - new_h
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw, dh = dw / 2, dh / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return new_h + top + bottom, new_w + left + right, new_h, new_w, top, left


def letterbox_into(img, dst, geom):
    """Write the letterboxed `img` into `dst` (out_h x out_w x 3, uint8).

    `dst` must already hold the grey padding outside the image area; only the
    image area is written.
    """
    _, _, new_h, new_w, top, left = geom
    h, w = img.shape[:2]
    view = dst[top:top + new_h, left:left + new_w]
    if (h, w) == (new_h, new_w):
        view[...] = img
    else:
        view[...] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
