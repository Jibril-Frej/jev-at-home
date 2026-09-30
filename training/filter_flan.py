"""Copy the FLAN ATP pool (prepare_flan.py output) without any Yelp task: the Yelp review data may not be used to train a
redistributed model. Keeps the row order, so the first N rows are still the same random sample minus the Yelp rows
(train_atp --n-examples N reads a prefix).

usage: filter_flan.py data/flan_atp data/flan_atp_ny
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC, DST = (Path(a) for a in (sys.argv[1:3] if len(sys.argv) > 2 else ("data/flan_atp", "data/flan_atp_ny")))
DST.mkdir(exist_ok=True)
for split in ("train_pool", "heldout"):
    meta = pq.read_table(SRC / f"{split}.meta.parquet")
    names = np.asarray(meta.column("task_name").to_pylist(), dtype=object)
    keep = np.array(["yelp" not in n.lower() for n in names])
    dropped = sorted(set(names[~keep].tolist()))
    tokens = np.load(SRC / f"{split}.tokens.npy", mmap_mode="r")
    offsets = np.load(SRC / f"{split}.offsets.npy")
    lengths = np.diff(offsets)[keep]
    new_offsets = np.zeros(len(lengths) + 1, dtype=offsets.dtype)
    np.cumsum(lengths, out=new_offsets[1:])
    out = np.lib.format.open_memmap(DST / f"{split}.tokens.npy", mode="w+", dtype=tokens.dtype, shape=(int(new_offsets[-1]),))
    # copy contiguous runs of kept rows
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
    pos = 0
    for a, b in zip(edges[::2], edges[1::2]):
        chunk = tokens[offsets[a]:offsets[b]]
        out[pos:pos + len(chunk)] = chunk
        pos += len(chunk)
    assert pos == new_offsets[-1]
    out.flush(); del out
    np.save(DST / f"{split}.offsets.npy", new_offsets)
    pq.write_table(meta.filter(pa.array(keep)), DST / f"{split}.meta.parquet")
    m = json.loads((SRC / f"{split}.meta.json").read_text())
    m.update({"n": int(keep.sum()), "total_tokens": int(new_offsets[-1]), "mean_length": float(lengths.mean()),
              "source": str(SRC), "dropped_tasks": dropped, "n_dropped": int((~keep).sum())})
    (DST / f"{split}.meta.json").write_text(json.dumps(m, indent=1))
    print(f"{split}: {len(keep):,} -> {int(keep.sum()):,} rows, dropped {int((~keep).sum()):,} from {dropped}", flush=True)
