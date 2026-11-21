"""Decode worker process.

Each worker takes whole videos (mp4 bytes) from a task queue, decodes them
with PyAV on one core, keeps K frames per second (first frame in each
1/K-second slot), letterboxes them exactly like Ultralytics and writes them
into "chunks" in a shared-memory block owned by the main process. It then
sends a small message describing the chunk; the main process copies it to the
GPU and hands the chunk back through `free_q`.

Sequential sampling (odd K): frames in even slots (0, 2, 4, ...) are
"primary" and always analysed. Frames in odd slots are "deferred": they go to
separate chunks, each tagged with how many primary frames its second has, and
the main process analyses them only if the primary frames of that second
disagree. When ceil(K/2) primary frames agree on every class and on the people
count, the majority/median over all K frames is already decided, so the
result is identical to analysing all K.

Messages are sent on this worker's own pipe (`conn`), synchronously from the
worker's main thread, so a crash cannot leave a half-sent message or a held
lock that blocks other workers:
  ("start", wid, vid)
  ("chunk", wid, chunk_id, vid, n, out_h, out_w, seconds, kind, n_primary)
      kind "p" (primary) or "d" (deferred); n_primary is a list for "d"
  ("end", wid, vid, last_second, n_decoded, error_or_None)
"""
import io
import math
import os
import traceback
from fractions import Fraction

import numpy as np

SLOT_BYTES = 640 * 640 * 3  # max letterboxed frame


class ChunkWriter:
    """Fills shared-memory chunks with letterboxed frames of one kind."""

    def __init__(self, kind, wid, vid, block, chunk_frames, free_q, msg_q, cv2, pad):
        self.kind, self.wid, self.vid = kind, wid, vid
        self.block, self.chunk_frames = block, chunk_frames
        self.free_q, self.msg_q, self.cv2, self.pad = free_q, msg_q, cv2, pad
        self.cid = None
        self.view = None
        self.geom = None
        self.geom_hw = None
        self.n = 0
        self.secs, self.extra = [], []

    def send(self):
        if self.cid is not None and self.n:
            g = self.geom
            self.msg_q.put(("chunk", self.wid, self.cid, self.vid, self.n, g[0], g[1],
                            self.secs, self.kind, self.extra if self.kind == "d" else None))
        elif self.cid is not None:
            self.free_q.put(self.cid)
        self.cid, self.view, self.n, self.secs, self.extra = None, None, 0, [], []

    def add(self, sec, img, geometry, extra=None, before_send=None):
        h, w = img.shape[:2]
        if self.geom_hw != (h, w):
            if self.n and before_send:
                before_send()
            self.send()  # never mix shapes in one chunk
            self.geom = geometry(h, w)
            self.geom_hw = (h, w)
        out_h, out_w, new_h, new_w, top, left = self.geom
        if self.cid is None:
            self.cid = self.free_q.get()
            self.view = self.block[self.cid, : self.chunk_frames * out_h * out_w * 3].reshape(
                self.chunk_frames, out_h, out_w, 3)
        dst = self.view[self.n]
        pad = self.pad
        if top:
            dst[:top] = pad
        if top + new_h < out_h:
            dst[top + new_h:] = pad
        if left:
            dst[:, :left] = pad
        if left + new_w < out_w:
            dst[:, left + new_w:] = pad
        area = dst[top:top + new_h, left:left + new_w]
        if (h, w) == (new_h, new_w):
            area[...] = img
        else:
            area[...] = self.cv2.resize(img, (new_w, new_h), interpolation=self.cv2.INTER_LINEAR)
        self.secs.append(sec)
        self.extra.append(extra)
        self.n += 1
        if self.n == self.chunk_frames:
            if before_send:
                before_send()
            self.send()


def worker_main(wid, task_q, conn, free_q, shm_name, n_chunks, chunk_frames, cfg):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import av
    import cv2
    from multiprocessing import shared_memory

    from .letterbox import PAD, geometry
    from .video import _rotation_cw, rotate

    cv2.setNumThreads(1)

    class _Sender:  # ChunkWriter expects .put()
        put = conn.send

    msg_q = _Sender()
    av.logging.set_level(av.logging.PANIC)  # damaged files: no log spam
    shm = shared_memory.SharedMemory(name=shm_name)
    block = np.ndarray((n_chunks, chunk_frames * SLOT_BYTES), np.uint8, shm.buf)
    k = cfg["per_second"]
    skip_nonref = cfg.get("skip_nonref", False)
    sequential = bool(cfg.get("sequential")) and k >= 3 and k % 2 == 1

    while True:
        task = task_q.get()
        if task is None:
            break
        vid, data = task
        msg_q.put(("start", wid, vid))
        if os.environ.get("VIDX_TEST_CRASH") == vid:  # test hook: simulate a decoder crash
            import signal
            os.kill(os.getpid(), signal.SIGSEGV)
        if os.environ.get("VIDX_TEST_HANG") == vid:  # test hook: simulate a hung decoder
            import time
            time.sleep(10 ** 6)
        prim = ChunkWriter("p", wid, vid, block, chunk_frames, free_q, msg_q, cv2, PAD)
        dfr = ChunkWriter("d", wid, vid, block, chunk_frames, free_q, msg_q, cv2, PAD)
        held = []  # deferred frames of the current second
        cur = {"sec": None, "n_primary": 0}

        def flush_second():
            # deferred frames of the finished second, tagged with its primaries;
            # primary chunk goes out first so the main process can decide.
            for s, img in held:
                dfr.add(s, img, geometry, extra=cur["n_primary"], before_send=prim.send)
            held.clear()

        last_second, n_decoded, error = -1, 0, None
        try:
            container = av.open(io.BytesIO(data), mode="r", metadata_errors="ignore")
            try:
                if not container.streams.video:
                    raise ValueError("no video stream")
                stream = container.streams.video[0]
                cc = stream.codec_context
                if skip_nonref:
                    cc.skip_frame = "NONREF"
                tb = stream.time_base or Fraction(1, 1000)
                fps = float(stream.average_rate or stream.guessed_rate or 30)
                first_pts, max_pkt_pts, rot, last_slot = None, None, None, None
                for packet in container.demux(stream):
                    if packet.pts is not None and (max_pkt_pts is None or packet.pts > max_pkt_pts):
                        max_pkt_pts = packet.pts
                    for frame in packet.decode():
                        idx = n_decoded
                        n_decoded += 1
                        if frame.pts is not None:
                            if first_pts is None:
                                first_pts = frame.pts
                            t = (frame.pts - first_pts) * tb
                        else:
                            t = Fraction(idx) / Fraction(fps)
                        if t < 0:
                            t = Fraction(0)
                        sec = math.floor(t)
                        if sec > last_second:
                            last_second = sec
                        slot_i = 0
                        if k:
                            slot_i = math.floor((t - sec) * k)
                            slot = (sec, slot_i)
                            if slot == last_slot:
                                continue
                            last_slot = slot
                        if rot is None:
                            rot = _rotation_cw(stream, frame)
                        img = rotate(frame.to_ndarray(format="bgr24"), rot)
                        if sequential:
                            if sec != cur["sec"]:
                                flush_second()
                                cur["sec"], cur["n_primary"] = sec, 0
                            if slot_i % 2:
                                held.append((sec, img))
                                continue
                            cur["n_primary"] += 1
                        prim.add(sec, img, geometry)
                if skip_nonref and first_pts is not None and max_pkt_pts is not None:
                    # skipped B-frames may lie after the last decoded frame
                    last_second = max(last_second, math.floor((max_pkt_pts - first_pts) * tb))
            finally:
                container.close()
        except Exception as e:
            error = f"could not decode video: {type(e).__name__}"
            if cfg.get("debug"):
                traceback.print_exc()
        try:
            flush_second()
        except Exception:
            held.clear()
        prim.send()
        dfr.send()
        if n_decoded == 0 and error is None:
            error = "no frames decoded"
        msg_q.put(("end", wid, vid, last_second, n_decoded, error))
    shm.close()
