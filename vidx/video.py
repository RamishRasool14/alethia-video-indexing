"""Decode one video (mp4 bytes) with PyAV.

Yields BGR frames with their second index, applying the rotation tag.
Timestamps are counted from the first decoded frame (brief, section 4).
"""
import io
import math
from fractions import Fraction

import av
import cv2
import numpy as np


def _rotation_cw(stream, frame):
    """Clockwise rotation in degrees (0/90/180/270) needed to display upright."""
    # PyAV >= 14 exposes the display-matrix rotation (counter-clockwise degrees,
    # same sign convention as FFmpeg's av_display_rotation_get).
    rot = getattr(frame, "rotation", None)
    if rot is not None:
        try:
            return int(round(-float(rot))) % 360
        except (TypeError, ValueError):
            pass
    # Older files / PyAV: the legacy "rotate" tag holds clockwise degrees.
    tag = (stream.metadata or {}).get("rotate")
    if tag is not None:
        try:
            return int(round(float(tag))) % 360
        except ValueError:
            pass
    return 0


def rotate(img, cw):
    cw = cw % 360
    if cw == 0:
        return img
    if cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img  # non-right angles: leave as is


class VideoReader:
    """Iterate over (second, bgr_frame) for the frames chosen for analysis.

    per_second: how many frames to analyse per second (0 = every frame).
    Frames are chosen by timestamp: each second is split into `per_second`
    equal slots and the first frame falling in each slot is used.

    After iteration, `last_second` is the second holding the last decoded
    frame (or -1 if no frame was decoded), and `n_decoded` counts frames.
    """

    def __init__(self, data, per_second=0, threads=None):
        self.data = data
        self.per_second = per_second
        self.threads = threads
        self.last_second = -1
        self.n_decoded = 0
        self.last_t = 0.0  # timestamp (s) of the frame most recently yielded

    def __iter__(self):
        container = av.open(io.BytesIO(self.data), mode="r")
        try:
            if not container.streams.video:
                raise ValueError("no video stream")
            stream = container.streams.video[0]
            if self.threads is not None:
                stream.thread_type = "AUTO"
                stream.thread_count = self.threads
            tb = stream.time_base or Fraction(1, 1000)
            fps = float(stream.average_rate or stream.guessed_rate or 30)
            first_pts = None
            rot = None
            last_slot = None
            k = self.per_second
            for frame in container.decode(stream):
                idx = self.n_decoded
                self.n_decoded += 1
                if frame.pts is not None:
                    if first_pts is None:
                        first_pts = frame.pts
                    t = (frame.pts - first_pts) * tb  # exact Fraction
                else:
                    t = Fraction(idx) / Fraction(fps)
                if t < 0:
                    t = Fraction(0)
                sec = math.floor(t)
                if sec > self.last_second:
                    self.last_second = sec
                if k:
                    slot = (sec, math.floor((t - sec) * k))
                    if slot == last_slot:
                        continue
                    last_slot = slot
                if rot is None:
                    rot = _rotation_cw(stream, frame)
                img = frame.to_ndarray(format="bgr24")
                self.last_t = float(t)
                yield sec, rotate(img, rot)
        finally:
            container.close()
