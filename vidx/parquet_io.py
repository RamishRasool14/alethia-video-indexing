"""Read videos (mp4 bytes) from FineVideo Parquet files, tolerating damage."""
import os

import pyarrow.parquet as pq


def list_parquet(folder):
    names = sorted(n for n in os.listdir(folder) if n.endswith(".parquet"))
    return [os.path.join(folder, n) for n in names]


def iter_videos(path):
    """Yield (video_id, mp4_bytes_or_None, error_or_None) for every row.

    video_id is "<file name>:<row>". If the file or a part of it cannot be
    read, yield an error for each affected row (or one for the whole file if
    even the row count is unknown).
    """
    name = os.path.basename(path)
    try:
        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows
        n_groups = pf.num_row_groups
    except Exception as e:  # unreadable file: one error line for row 0
        yield f"{name}:0", None, f"could not read parquet file: {type(e).__name__}"
        return
    row = 0
    for g in range(n_groups):
        g_rows = pf.metadata.row_group(g).num_rows
        try:
            table = pf.read_row_group(g, columns=["mp4"])
            col = table.column("mp4")
        except Exception as e:
            for r in range(row, row + g_rows):
                yield f"{name}:{r}", None, f"could not read parquet rows: {type(e).__name__}"
            row += g_rows
            continue
        for i in range(len(col)):
            v = col[i].as_py()
            if not v:
                yield f"{name}:{row}", None, "empty mp4 bytes"
            else:
                yield f"{name}:{row}", v, None
            row += 1
        del table, col
    if row < n_rows:  # metadata says more rows than the groups held
        for r in range(row, n_rows):
            yield f"{name}:{r}", None, "row missing from parquet file"
