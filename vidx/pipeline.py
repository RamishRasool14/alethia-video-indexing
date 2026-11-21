"""Fast pipeline: parallel decode workers -> shared memory -> one GPU loop.

Main process:
  - a reader thread reads Parquet files in order and puts (video id, mp4
    bytes) on the task queue (error rows go straight to the output);
  - N worker processes decode/sample/letterbox into shared-memory chunks
    (see worker.py);
  - the main thread batches frames of the same shape, runs YOLO on the GPU,
    counts the five classes per frame, decides which deferred frames are
    needed (sequential sampling), and writes each video's lines as soon as it
    is complete;
  - a worker that dies (e.g. a decoder crash on a damaged file) is replaced,
    and its current video gets an error line.
"""
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from multiprocessing import shared_memory

import numpy as np

from .aggregate import per_second
from .parquet_io import iter_videos, list_parquet
from .worker import SLOT_BYTES, worker_main


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class VideoState:
    __slots__ = ("secs", "counts", "chunks_sent", "chunks_done", "end", "prim", "waiting")

    def __init__(self):
        self.secs, self.counts = [], []
        self.chunks_sent = self.chunks_done = 0
        self.end = None  # (last_second, n_decoded, error)
        self.prim = defaultdict(list)  # second -> [(presence bytes, people)] of primaries
        self.waiting = []  # deferred chunks not yet fully decided


def records_for(vid, st):
    last_second, _, error = st.end
    recs = []
    if last_second >= 0:
        secs = np.concatenate(st.secs) if st.secs else np.zeros(0, np.int64)
        counts = np.concatenate(st.counts) if st.counts else np.zeros((0, 5), np.int32)
        for s, (objects, people) in enumerate(per_second(secs, counts, last_second)):
            recs.append({"video": vid, "second": s, "objects": objects, "people": people})
    if error:
        recs.append({"video": vid, "error": error})
    return recs


class Deferred:
    """A deferred chunk waiting for its seconds' primary results."""
    __slots__ = ("msg", "need", "decided", "n_undecided")

    def __init__(self, msg):
        self.msg = msg
        n = msg[4]
        self.need = [None] * n
        self.n_undecided = n


def run(input_dir, output_path, model_factory, n_workers, per_second_k=3, skip_nonref=False,
        sequential=True, max_batch=64, chunk_frames=32, chunks_per_worker=6, debug=False):
    import torch

    from .postprocess import count_classes

    # forkserver: workers are forked from a clean single-threaded server, so a
    # worker restarted mid-run does not inherit CUDA or thread locks.
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["av", "cv2", "numpy", "vidx.worker"])
    t0 = time.time()
    out = open(output_path, "w")
    need_primary = (per_second_k + 1) // 2
    cfg = {"per_second": per_second_k, "skip_nonref": skip_nonref, "debug": debug,
           "sequential": sequential}
    task_q = ctx.Queue(maxsize=max(4, n_workers))
    msg_q = ctx.Queue()
    workers = {}  # wid -> dict(proc, free_q, shm, block, current)

    def start_worker(wid):
        shm = shared_memory.SharedMemory(create=True,
                                         size=chunks_per_worker * chunk_frames * SLOT_BYTES)
        block = np.ndarray((chunks_per_worker, chunk_frames * SLOT_BYTES), np.uint8, shm.buf)
        free_q = ctx.SimpleQueue()
        for c in range(chunks_per_worker):
            free_q.put(c)
        p = ctx.Process(target=worker_main, daemon=True,
                        args=(wid, task_q, msg_q, free_q, shm.name, chunks_per_worker,
                              chunk_frames, cfg))
        p.start()
        workers[wid] = dict(proc=p, free_q=free_q, shm=shm, block=block, current=None)

    videos = {}  # vid -> VideoState
    reader_done = threading.Event()
    n_tasks = [0]
    direct_errors = queue.Queue()  # rows that failed before decoding

    def reader():
        try:
            for path in list_parquet(input_dir):
                for vid, data, err in iter_videos(path):
                    if err:
                        direct_errors.put({"video": vid, "error": err})
                        continue
                    n_tasks[0] += 1
                    task_q.put((vid, data))
                if debug:
                    log(f"[{time.time() - t0:7.1f}s] queued {os.path.basename(path)}")
        except Exception as e:  # should not happen; keep going with what we have
            log(f"reader error: {type(e).__name__}: {e}")
        finally:
            reader_done.set()

    # Start reading and decoding first; load the model while they work.
    rt = threading.Thread(target=reader, daemon=True)
    rt.start()
    for wid in range(n_workers):
        start_worker(wid)
    model = model_factory()
    log(f"[{time.time() - t0:7.1f}s] model ready")

    pending = defaultdict(list)  # (h, w) -> items (wid, cid, idx|None, n, vid, secs, kind, ref)
    n_pending = defaultdict(int)
    stats = {"finished": 0, "frames": 0, "deferred_total": 0, "deferred_run": 0}
    last_report = time.time()

    def finalize_ready(vid):
        st = videos.get(vid)
        if st is not None and st.end is not None and st.chunks_done == st.chunks_sent:
            out.write("".join(json.dumps(r) + "\n" for r in records_for(vid, st)))
            del videos[vid]
            stats["finished"] += 1

    def enqueue(item, h, w):
        pending[(h, w)].append(item)
        n_pending[(h, w)] += item[3]

    def try_decide(st, d):
        """Decide deferred frames whose seconds have all their primaries."""
        _, wid, cid, vid, n, h, w, secs, _, n_prim = d.msg
        for i in range(n):
            if d.need[i] is not None:
                continue
            got = st.prim.get(secs[i], ())
            if n_prim[i] < need_primary:
                d.need[i] = True
            elif len(got) >= n_prim[i]:
                first = got[0]
                d.need[i] = any(g != first for g in got)
            else:
                continue
            d.n_undecided -= 1
        if d.n_undecided:
            return False
        idx = np.flatnonzero(d.need)
        stats["deferred_total"] += n
        stats["deferred_run"] += len(idx)
        if len(idx):
            enqueue((wid, cid, idx, len(idx), vid, [secs[i] for i in idx], "d", None), h, w)
        else:
            workers[wid]["free_q"].put(cid)
            st.chunks_done += 1
        return True

    def handle(msg):
        kind = msg[0]
        if kind == "start":
            _, wid, vid = msg
            workers[wid]["current"] = vid
            videos.setdefault(vid, VideoState())
        elif kind == "chunk":
            _, wid, cid, vid, n, h, w, secs, ckind, _ = msg
            st = videos[vid]
            st.chunks_sent += 1
            if ckind == "p":
                enqueue((wid, cid, None, n, vid, secs, "p", None), h, w)
            else:
                d = Deferred(msg)
                if not try_decide(st, d):
                    st.waiting.append(d)
        elif kind == "end":
            _, wid, vid, last_second, n_decoded, error = msg
            workers[wid]["current"] = None
            videos[vid].end = (last_second, n_decoded, error)
            finalize_ready(vid)

    def run_batch(shape):
        h, w = shape
        items, total = [], 0
        lst = pending[shape]
        while lst and total < max_batch:
            it = lst.pop(0)
            items.append(it)
            total += it[3]
        n_pending[shape] -= total
        parts = []
        for (wid, cid, idx, n, vid, secs, kind, _) in items:
            view = workers[wid]["block"][cid]
            if idx is None:
                arr = view[: n * h * w * 3].reshape(n, h, w, 3)
            else:
                arr = view[: (int(idx[-1]) + 1) * h * w * 3].reshape(-1, h, w, 3)[idx]
            parts.append(torch.from_numpy(arr).to("cuda"))
            workers[wid]["free_q"].put(cid)
        x = torch.cat(parts) if len(parts) > 1 else parts[0]
        x = x.permute(0, 3, 1, 2).flip(1).float().div_(255)  # BGR->RGB, 0..1
        counts = []
        for i in range(0, x.shape[0], model.max_batch):
            counts.append(count_classes(model(x[i:i + model.max_batch])))
        counts = torch.cat(counts).cpu().numpy()
        off = 0
        touched = set()
        for (wid, cid, idx, n, vid, secs, kind, _) in items:
            st = videos[vid]
            c = counts[off:off + n]
            off += n
            st.secs.append(np.asarray(secs, np.int64))
            st.counts.append(c)
            st.chunks_done += 1
            if kind == "p":
                pres = c > 0
                for j, s in enumerate(secs):
                    st.prim[s].append((pres[j].tobytes(), int(c[j, 0])))
                touched.add(vid)
            else:
                finalize_ready(vid)
        for vid in touched:
            st = videos.get(vid)
            if st is None:  # already complete and written in the loop above
                continue
            if st.waiting:
                st.waiting = [d for d in st.waiting if not try_decide(st, d)]
            finalize_ready(vid)
        stats["frames"] += total

    def check_workers():
        for wid, wk in list(workers.items()):
            if wk["proc"].is_alive():
                continue
            vid = wk["current"]
            log(f"worker {wid} died (exit {wk['proc'].exitcode}) on {vid}; restarting")
            # finish the work it already handed over (its memory is ours)
            for st in list(videos.values()):
                for d in st.waiting:
                    if d.msg[1] == wid:
                        d.need = [True if x is None else x for x in d.need]
                        d.n_undecided = 0
                        try_decide(st, d)
                st.waiting = [d for d in st.waiting if d.n_undecided]
            while True:
                shapes = [sh for sh, lst in pending.items() if any(it[0] == wid for it in lst)]
                if not shapes:
                    break
                run_batch(shapes[0])
            if vid is not None and vid in videos:
                st = videos[vid]
                secs = np.concatenate(st.secs) if st.secs else np.zeros(0, np.int64)
                st.end = (int(secs.max()) if len(secs) else -1, 0,
                          "could not decode video: decoder crashed")
                finalize_ready(vid)
            wk["shm"].close()
            wk["shm"].unlink()
            del workers[wid]
            start_worker(wid)

    try:
        while True:
            while not direct_errors.empty():
                out.write(json.dumps(direct_errors.get()) + "\n")
            got = False
            try:
                msg = msg_q.get(timeout=0.002 if any(n_pending.values()) else 0.05)
                handle(msg)
                got = True
                while True:  # drain without blocking
                    handle(msg_q.get_nowait())
            except queue.Empty:
                pass
            # run a batch if any shape has a full batch, or if nothing new arrived
            ready = [s for s, n in n_pending.items() if n >= max_batch]
            if not ready and not got:
                ready = [s for s, n in n_pending.items() if n > 0]
            if ready:
                run_batch(max(ready, key=lambda s: n_pending[s]))
            elif not got:
                check_workers()
                if reader_done.is_set() and direct_errors.empty() \
                        and stats["finished"] >= n_tasks[0] and not videos:
                    break
            if debug and time.time() - last_report > 30:
                el = time.time() - t0
                log(f"[{el:7.1f}s] videos done {stats['finished']}, frames {stats['frames']} "
                    f"({stats['frames'] / el:.0f}/s)")
                last_report = time.time()
    finally:
        task_q.cancel_join_thread()  # never block exit on unread tasks
        for _ in workers:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
        for wk in workers.values():
            wk["proc"].join(timeout=5)
            if wk["proc"].is_alive():
                wk["proc"].kill()
            wk["shm"].close()
            try:
                wk["shm"].unlink()
            except FileNotFoundError:
                pass
        out.close()
    log(f"done: {stats['finished']} videos, {stats['frames']} frames analysed "
        f"(deferred run {stats['deferred_run']}/{stats['deferred_total']}) "
        f"in {time.time() - t0:.1f}s")
