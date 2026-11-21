"""Fast pipeline: parallel decode workers -> shared memory -> GPU threads.

Main process, main thread:
  - starts a reader thread that reads Parquet files in order and puts
    (video id, mp4 bytes) on the task queue (rows that cannot be read go
    straight to the output as error lines);
  - starts N decode worker processes (worker.py) that sample, letterbox and
    write frames into shared-memory chunks;
  - routes worker messages, forms batches of same-shape frames, decides which
    deferred frames are needed (sequential sampling), and writes each video's
    lines as soon as it is complete;
  - replaces a worker that dies (e.g. decoder crash) and writes an error line
    for the video it was on.
Main process, inference threads (2 by default):
  - each has its own CUDA stream and TensorRT execution context, so one batch
    is copied/pre-processed while another runs: the GPU never waits for
    Python bookkeeping;
  - copies frames from pinned shared memory, converts BGR uint8 -> RGB float,
    runs YOLO11l, counts the five classes per frame (GPU NMS) and returns the
    counts.
"""
import json
import multiprocessing as mp
import multiprocessing.connection as mpc
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
    __slots__ = ("wk", "msg", "need", "n_undecided")

    def __init__(self, wk, msg):
        self.wk, self.msg = wk, msg
        self.need = [None] * msg[4]
        self.n_undecided = msg[4]


class Item:
    """Frames of one chunk (all, or a subset `idx`) waiting for the GPU."""
    __slots__ = ("wk", "cid", "idx", "n", "vid", "secs", "kind")

    def __init__(self, wk, cid, idx, n, vid, secs, kind):
        self.wk, self.cid, self.idx, self.n = wk, cid, idx, n
        self.vid, self.secs, self.kind = vid, secs, kind


def _pin(block):
    """Page-lock a shared-memory block so host->GPU copies are fast DMA."""
    import torch

    try:
        rc = torch.cuda.cudart().cudaHostRegister(block.ctypes.data, block.nbytes, 0)
        return int(rc) == 0 if not hasattr(rc, "value") else rc.value == 0
    except Exception:
        return False


def _unpin(block):
    import torch

    try:
        torch.cuda.cudart().cudaHostUnregister(block.ctypes.data)
    except Exception:
        pass


def run(input_dir, output_path, model_factory, n_workers, per_second_k=3, skip_nonref=False,
        sequential=True, max_batch=64, chunk_frames=32, chunks_per_worker=6,
        gpu_threads=2, stall_timeout=120.0, debug=False):
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
    workers = {}  # wid -> dict(proc, free_q, shm, block, current, pinned)
    retired = []  # shared memory of dead workers, freed at the end
    cuda_ready = [False]

    def start_worker(wid):
        shm = shared_memory.SharedMemory(create=True,
                                         size=chunks_per_worker * chunk_frames * SLOT_BYTES)
        block = np.ndarray((chunks_per_worker, chunk_frames * SLOT_BYTES), np.uint8, shm.buf)
        free_q = ctx.SimpleQueue()
        for c in range(chunks_per_worker):
            free_q.put(c)
        recv_end, send_end = ctx.Pipe(duplex=False)
        p = ctx.Process(target=worker_main, daemon=True,
                        args=(wid, task_q, send_end, free_q, shm.name, chunks_per_worker,
                              chunk_frames, cfg))
        p.start()
        send_end.close()  # only the worker writes; EOF on recv_end = worker gone
        wk = dict(wid=wid, proc=p, free_q=free_q, shm=shm, block=block, current=None,
                  since=None, pinned=False, conn=recv_end, eof=False)
        if cuda_ready[0]:
            wk["pinned"] = _pin(block)
        workers[wid] = wk

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
    cuda_ready[0] = True
    for wk in workers.values():
        wk["pinned"] = _pin(wk["block"])
    log(f"[{time.time() - t0:7.1f}s] model ready; pinned {sum(w['pinned'] for w in workers.values())}"
        f"/{len(workers)} buffers")

    # ------------------------------------------------------------ GPU threads
    batch_q = queue.Queue()
    result_q = queue.Queue()

    def infer_loop():
        stream = torch.cuda.Stream()
        runner = model.runner()
        with torch.cuda.stream(stream), torch.inference_mode():
            while True:
                job = batch_q.get()
                if job is None:
                    break
                items, h, w = job
                try:
                    parts = []
                    for it in items:
                        view = it.wk["block"][it.cid]
                        if it.idx is None:
                            arr = view[: it.n * h * w * 3].reshape(it.n, h, w, 3)
                        else:
                            arr = view[: (int(it.idx[-1]) + 1) * h * w * 3].reshape(
                                -1, h, w, 3)[it.idx]
                        parts.append(torch.from_numpy(arr).to("cuda", non_blocking=True))
                    x = torch.cat(parts) if len(parts) > 1 else parts[0]
                    copied = torch.cuda.Event()
                    copied.record(stream)
                    x = x.permute(0, 3, 1, 2).flip(1).float().div_(255)  # BGR->RGB, 0..1
                    counts = []
                    for i in range(0, x.shape[0], model.max_batch):
                        counts.append(count_classes(runner(x[i:i + model.max_batch])))
                    copied.synchronize()
                    counts = torch.cat(counts).cpu().numpy()
                    result_q.put((items, counts, None))
                except Exception as e:  # report, never kill the thread
                    result_q.put((items, None, e))

    threads = [threading.Thread(target=infer_loop, daemon=True) for _ in range(gpu_threads)]
    for t in threads:
        t.start()

    pending = defaultdict(list)  # (h, w) -> [Item]
    n_pending = defaultdict(int)
    in_flight = [0]
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
        n_pending[(h, w)] += item.n

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
            enqueue(Item(d.wk, cid, idx, len(idx), vid, [secs[i] for i in idx], "d"), h, w)
        else:
            d.wk["free_q"].put(cid)
            st.chunks_done += 1
        return True

    def handle(msg):
        kind = msg[0]
        if kind == "start":
            _, wid, vid = msg
            workers[wid]["current"] = vid
            workers[wid]["since"] = time.time()
            videos.setdefault(vid, VideoState())
        elif kind == "chunk":
            _, wid, cid, vid, n, h, w, secs, ckind, _ = msg
            workers[wid]["since"] = time.time()
            st = videos[vid]
            st.chunks_sent += 1
            wk = workers[wid]
            if ckind == "p":
                enqueue(Item(wk, cid, None, n, vid, secs, "p"), h, w)
            else:
                d = Deferred(wk, msg)
                if not try_decide(st, d):
                    st.waiting.append(d)
        elif kind == "end":
            _, wid, vid, last_second, n_decoded, error = msg
            workers[wid]["current"] = None
            videos[vid].end = (last_second, n_decoded, error)
            finalize_ready(vid)

    def submit(shape):
        lst = pending[shape]
        items, total = [], 0
        while lst and total < max_batch:
            it = lst.pop(0)
            items.append(it)
            total += it.n
        n_pending[shape] -= total
        in_flight[0] += 1
        batch_q.put((items, shape[0], shape[1]))

    def handle_result(res):
        items, counts, err = res
        in_flight[0] -= 1
        if err is not None:
            raise err
        touched = set()
        off = 0
        for it in items:
            it.wk["free_q"].put(it.cid)
            st = videos.get(it.vid)
            c = counts[off:off + it.n]
            off += it.n
            stats["frames"] += it.n
            if st is None:
                continue
            st.secs.append(np.asarray(it.secs, np.int64))
            st.counts.append(c)
            st.chunks_done += 1
            if it.kind == "p":
                pres = c > 0
                for j, s in enumerate(it.secs):
                    st.prim[s].append((pres[j].tobytes(), int(c[j, 0])))
                touched.add(it.vid)
            else:
                finalize_ready(it.vid)
        for vid in touched:
            st = videos.get(vid)
            if st is None:  # already complete and written above
                continue
            if st.waiting:
                st.waiting = [d for d in st.waiting if not try_decide(st, d)]
            finalize_ready(vid)

    def drain_results(block=False):
        got = False
        while True:
            try:
                res = result_q.get(timeout=0.5) if block and not got else result_q.get_nowait()
            except queue.Empty:
                return got
            handle_result(res)
            got = True

    def poll_messages(timeout):
        """Handle all available worker messages; True if any arrived."""
        conns = {wk["conn"]: wk for wk in workers.values() if not wk["eof"]}
        if not conns:
            time.sleep(timeout)
            return False
        got = False
        for c in mpc.wait(list(conns), timeout=timeout):
            wk = conns[c]
            try:
                while c.poll():
                    handle(c.recv())
                    got = True
            except (EOFError, OSError):
                wk["eof"] = True  # worker exited; check_workers handles it
        return got

    def check_workers():
        for wid, wk in list(workers.items()):
            # stuck decoder: no output from this worker for stall_timeout seconds
            hung = wk["current"] is not None and time.time() - wk["since"] > stall_timeout
            if hung:  # a decoder stuck on one video: treat it like a crash
                wk["proc"].kill()
                wk["proc"].join(timeout=5)
            if wk["proc"].is_alive():
                continue
            try:  # read everything it sent before it died
                while wk["conn"].poll():
                    handle(wk["conn"].recv())
            except (EOFError, OSError):
                pass
            vid = wk["current"]
            reason = "timed out" if hung else "crashed"
            log(f"worker {wid} {reason} (exit {wk['proc'].exitcode}) on {vid}; restarting")
            # its deferred chunks can no longer be completed: analyse all frames
            for st in list(videos.values()):
                keep = []
                for d in st.waiting:
                    if d.wk is wk:
                        d.need = [True if x is None else x for x in d.need]
                        d.n_undecided = 0
                        try_decide(st, d)
                    else:
                        keep.append(d)
                st.waiting = keep
            # wait until everything it handed over has been analysed
            while any(it.wk is wk for lst in pending.values() for it in lst) or in_flight[0]:
                for sh, lst in list(pending.items()):
                    if lst:
                        submit(sh)
                drain_results(block=True)
            if vid is not None and vid in videos:
                st = videos[vid]
                secs = np.concatenate(st.secs) if st.secs else np.zeros(0, np.int64)
                st.end = (int(secs.max()) if len(secs) else -1, 0,
                          f"could not decode video: decoder {reason}")
                finalize_ready(vid)
            retired.append(wk)
            del workers[wid]
            start_worker(wid)

    max_in_flight = gpu_threads + 1
    last_check = time.time()
    try:
        while True:
            while not direct_errors.empty():
                out.write(json.dumps(direct_errors.get()) + "\n")
            got = drain_results()
            timeout = 0.001 if (any(n_pending.values()) or in_flight[0]) else 0.05
            got = poll_messages(timeout) or got
            # keep the GPU threads fed: full batches first; partial batches
            # only when the GPU would otherwise go idle
            while in_flight[0] < max_in_flight:
                ready = [s for s, n in n_pending.items() if n >= max_batch]
                if not ready and in_flight[0] < gpu_threads:
                    ready = [s for s, n in n_pending.items() if n > 0]
                if not ready:
                    break
                submit(max(ready, key=lambda s: n_pending[s]))
                got = True
            if not got or time.time() - last_check > 1.0:
                check_workers()
                last_check = time.time()
            if not got:
                if reader_done.is_set() and direct_errors.empty() and not in_flight[0] \
                        and stats["finished"] >= n_tasks[0] and not videos:
                    break
            if debug and time.time() - last_report > 30:
                el = time.time() - t0
                log(f"[{el:7.1f}s] videos done {stats['finished']}, frames {stats['frames']} "
                    f"({stats['frames'] / el:.0f}/s) in flight {in_flight[0]} "
                    f"pending {sum(n_pending.values())} tasks queued {task_q.qsize()}")
                last_report = time.time()
    finally:
        task_q.cancel_join_thread()  # never block exit on unread tasks
        for _ in threads:
            batch_q.put(None)
        for _ in workers:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
        for t in threads:
            t.join(timeout=10)
        for wk in list(workers.values()) + retired:
            wk["proc"].join(timeout=5)
            if wk["proc"].is_alive():
                wk["proc"].kill()
            if wk["pinned"]:
                _unpin(wk["block"])
            del wk["block"]
            wk["conn"].close()
            try:
                wk["shm"].close()
            except BufferError:  # a view may still be referenced; unlink anyway
                pass
            try:
                wk["shm"].unlink()
            except FileNotFoundError:
                pass
        out.close()
    log(f"done: {stats['finished']} videos, {stats['frames']} frames analysed "
        f"(deferred run {stats['deferred_run']}/{stats['deferred_total']}) "
        f"in {time.time() - t0:.1f}s")
