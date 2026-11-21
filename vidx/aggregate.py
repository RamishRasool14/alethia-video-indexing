"""Turn per-frame detection counts into per-second output lines.

Rules (brief, section 5):
- an object is present in a second if it is found in at least half of that
  second's frames;
- the people count is the median of the per-frame person counts, rounded down
  when it falls halfway between two integers.

We apply the same rules to whichever frames of a second were run through the
model (all of them, or a sample).
"""
import numpy as np

from . import CLASS_NAMES, PERSON_COL


def per_second(seconds, counts, last_second):
    """Aggregate per-frame counts into one record per second.

    seconds: int array (n,) second index of each analysed frame.
    counts:  int array (n, 5) detections per class per analysed frame.
    last_second: index of the second holding the last decoded frame.

    Returns a list of (objects, people) tuples, one per second 0..last_second.
    """
    out = []
    seconds = np.asarray(seconds, dtype=np.int64)
    counts = np.asarray(counts).reshape(len(seconds), len(CLASS_NAMES))
    if len(seconds):
        order = np.argsort(seconds, kind="stable")
        seconds = seconds[order]
        counts = counts[order]
    bounds = np.searchsorted(seconds, np.arange(last_second + 2))
    for s in range(last_second + 1):
        c = counts[bounds[s]:bounds[s + 1]]
        n = len(c)
        if n == 0:
            out.append(([], 0))
            continue
        present = (c > 0).sum(axis=0) * 2 >= n
        objects = [CLASS_NAMES[j] for j in range(len(CLASS_NAMES)) if present[j]]
        p = np.sort(c[:, PERSON_COL])
        if n % 2:
            people = int(p[n // 2])
        else:
            people = int((int(p[n // 2 - 1]) + int(p[n // 2])) // 2)
        out.append((objects, people))
    return out
