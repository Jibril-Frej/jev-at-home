"""Build the Answer Token Prediction training sets from FLAN 2022 (Open-Orca/FLAN parquet dump).

Follows "It's All in The [MASK]" (Clavié, Cooper, Warner 2025, arXiv 2502.03793), section 2.3:
  1. keep only examples whose target is a single token (here: " " + target is one ModernBERT BPE token,
     because the answer follows the "[unused0]" anchor after a space),
  2. hold out MMLU / BBH style tasks (FLAN already excludes them; we filter by name defensively),
  3. cap the number of examples per task so a handful of large datasets do not dominate,
  4. down-sample to the requested pool size, shuffle, and expose nested subsets
     (the 2M set is a prefix of the 5M set, which is a prefix of the 20M set).
  5. tokenize:  [CLS] <inputs> " " [unused0] <target> [SEP]   -> flat uint16 token stream + offsets.

Stages (run in order, each resumable):  filter -> sample -> tokenize
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MODEL = "answerdotai/ModernBERT-base"
HOLDOUT_RE = re.compile(r"mmlu|hendrycks|bbh|big_?bench", re.I)
SPACE_ID, ANCHOR_ID = 209, 50285  # " " and "[unused0]" in the ModernBERT vocabulary (asserted at run time)

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL)
        assert _tok.encode(" [unused0]", add_special_tokens=False) == [SPACE_ID, ANCHOR_ID]
    return _tok


# ----------------------------------------------------------------------------- stage 1: filter
def filter_file(src: str, dst: str) -> dict:
    tok = tokenizer()
    cache: dict[str, int | None] = {}
    pf = pq.ParquetFile(src)
    n_in = n_out = n_holdout = 0
    out_cols = {"inputs": [], "target": [], "target_id": [], "task_name": [], "template_type": [], "task_source": []}
    for batch in pf.iter_batches(batch_size=50_000, columns=["inputs", "targets", "_task_name", "_template_type", "_task_source"]):
        d = batch.to_pydict()
        for inp, tgt, name, ttype, tsrc in zip(d["inputs"], d["targets"], d["_task_name"], d["_template_type"], d["_task_source"]):
            n_in += 1
            if not tgt or not inp:
                continue
            t = tgt.strip()
            if not t or "\n" in t:
                continue
            tid = cache.get(t, -1)
            if tid == -1:
                ids = tok.encode(" " + t, add_special_tokens=False)
                tid = ids[0] if len(ids) == 1 else None
                cache[t] = tid
            if tid is None:
                continue
            if name and HOLDOUT_RE.search(name):
                n_holdout += 1
                continue
            out_cols["inputs"].append(inp.rstrip())
            out_cols["target"].append(t)
            out_cols["target_id"].append(tid)
            out_cols["task_name"].append(name or "")
            out_cols["template_type"].append(ttype or "")
            out_cols["task_source"].append(tsrc or "")
            n_out += 1
    table = pa.table({
        "inputs": pa.array(out_cols["inputs"], pa.string()),
        "target": pa.array(out_cols["target"], pa.string()),
        "target_id": pa.array(out_cols["target_id"], pa.int32()),
        "task_name": pa.array(out_cols["task_name"], pa.string()).dictionary_encode(),
        "template_type": pa.array(out_cols["template_type"], pa.string()).dictionary_encode(),
        "task_source": pa.array(out_cols["task_source"], pa.string()).dictionary_encode(),
    })
    tmp = dst + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, dst)
    return {"src": src, "n_in": n_in, "n_single_token": n_out, "n_holdout": n_holdout}


def stage_filter(args):
    files = sorted(glob.glob(os.path.join(args.raw, "*_data", "*.parquet")))
    if not files:
        sys.exit(f"no parquet files under {args.raw}")
    fdir = Path(args.out) / "filtered"
    fdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in files:
        name = Path(f).parent.name.replace("_data", "") + "__" + Path(f).stem + ".parquet"
        dst = fdir / name
        if not dst.exists():
            jobs.append((f, str(dst)))
    print(f"[filter] {len(files)} raw files, {len(jobs)} to do, workers={args.workers}", flush=True)
    stats_path = Path(args.out) / "filter_stats.jsonl"
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(args.workers) as ex, open(stats_path, "a") as fh:
        futs = [ex.submit(filter_file, s, d) for s, d in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            fh.write(json.dumps(r) + "\n"); fh.flush()
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[filter] {done}/{len(jobs)} files  {time.time()-t0:.0f}s  last: {Path(r['src']).parent.name} "
                      f"{r['n_single_token']}/{r['n_in']}", flush=True)
    tot = Counter()
    for line in open(stats_path):
        r = json.loads(line)
        tot["n_in"] += r["n_in"]; tot["n_single_token"] += r["n_single_token"]; tot["n_holdout"] += r["n_holdout"]
    print(f"[filter] total rows={tot['n_in']:,} single-token={tot['n_single_token']:,} holdout={tot['n_holdout']:,}", flush=True)


# ----------------------------------------------------------------------------- stage 2: sample
def stage_sample(args):
    fdir = Path(args.out) / "filtered"
    files = sorted(fdir.glob("*.parquet"))
    rng = np.random.default_rng(args.seed)
    # global row index = (file idx, row idx); load only task names
    task_arrays, sizes = [], []
    for f in files:
        col = pq.read_table(f, columns=["task_name"]).column("task_name")
        task_arrays.append(col.to_numpy(zero_copy_only=False).astype(object) if col.type == pa.string()
                           else np.asarray(col.combine_chunks().to_pylist(), dtype=object))
        sizes.append(len(col))
    tasks = np.concatenate(task_arrays)
    n_total = len(tasks)
    counts = Counter(tasks.tolist())
    print(f"[sample] {n_total:,} single-token examples across {len(counts)} task names from {len(files)} files", flush=True)
    pool_target = max(args.sizes) + args.n_heldout

    # per-task cap: the smallest cap such that sum(min(count, cap)) >= pool_target
    vals = np.array(sorted(counts.values()))
    lo, hi = 1, int(vals.max())
    def kept(cap): return int(np.minimum(vals, cap).sum())
    if kept(hi) < pool_target:
        cap = hi
        print(f"[sample] WARNING only {kept(hi):,} examples available, below the {pool_target:,} target", flush=True)
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if kept(mid) >= pool_target:
                hi = mid
            else:
                lo = mid + 1
        cap = lo
    print(f"[sample] per-task cap={cap:,} -> {kept(cap):,} examples", flush=True)

    # sample up to `cap` rows per task, uniformly at random
    order = rng.permutation(n_total)
    seen = Counter()
    chosen = np.zeros(n_total, dtype=bool)
    for idx in order:
        t = tasks[idx]
        if seen[t] < cap:
            seen[t] += 1
            chosen[idx] = True
    pool = np.flatnonzero(chosen)
    pool = pool[rng.permutation(len(pool))]
    # trim to exactly the target if we have more
    if len(pool) > pool_target:
        pool = pool[:pool_target]
    heldout = pool[:args.n_heldout]
    train_pool = pool[args.n_heldout:]
    print(f"[sample] pool={len(pool):,} train={len(train_pool):,} heldout={len(heldout):,}", flush=True)

    # map global row index -> (file, row)
    bounds = np.cumsum([0] + sizes)
    def split(gidx):
        f = np.searchsorted(bounds, gidx, side="right") - 1
        return f, gidx - bounds[f]
    np.save(Path(args.out) / "train_pool_rows.npy", np.stack(split(train_pool), axis=1).astype(np.int64))
    np.save(Path(args.out) / "heldout_rows.npy", np.stack(split(heldout), axis=1).astype(np.int64))
    meta = {
        "n_single_token_total": int(n_total), "n_task_names": len(counts), "per_task_cap": int(cap),
        "pool": int(len(pool)), "train_pool": int(len(train_pool)), "heldout": int(len(heldout)),
        "sizes": sorted(args.sizes), "seed": args.seed, "files": [f.name for f in files],
        "train_task_counts": dict(Counter(tasks[train_pool].tolist()).most_common()),
    }
    (Path(args.out) / "sample_meta.json").write_text(json.dumps(meta, indent=1))
    print("[sample] top tasks in train pool:", Counter(tasks[train_pool].tolist()).most_common(10), flush=True)


# ----------------------------------------------------------------------------- stage 3: tokenize
def tokenize_rows(file_path: str, rows: np.ndarray, max_len: int, head: int):
    """Return (list of token id arrays, list of target ids, task names) for the given rows of one filtered file."""
    tok = tokenizer()
    t = pq.read_table(file_path, columns=["inputs", "target_id", "task_name", "template_type"]).take(pa.array(rows))
    inputs = t.column("inputs").to_pylist()
    target_ids = t.column("target_id").to_pylist()
    names = t.column("task_name").to_pylist()
    ttypes = t.column("template_type").to_pylist()
    cls, sep = tok.cls_token_id, tok.sep_token_id
    budget = max_len - 5  # CLS, space, anchor, target, SEP
    enc = tok(inputs, add_special_tokens=False)["input_ids"]
    seqs = []
    n_trunc = 0
    for ids, tid in zip(enc, target_ids):
        if len(ids) > budget:
            ids = ids[:head] + ids[-(budget - head):]
            n_trunc += 1
        seqs.append(np.array([cls] + ids + [SPACE_ID, ANCHOR_ID, tid, sep], dtype=np.uint16))
    return seqs, names, ttypes, n_trunc


def _tok_job(a):
    return tokenize_rows(*a)


def stage_tokenize(args):
    fdir = Path(args.out) / "filtered"
    files = sorted(fdir.glob("*.parquet"))
    for split in ("train_pool", "heldout"):
        rows = np.load(Path(args.out) / f"{split}_rows.npy")
        out_prefix = Path(args.out) / split
        if (out_prefix.with_suffix(".meta.json")).exists():
            print(f"[tokenize] {split} already done", flush=True)
            continue
        n = len(rows)
        print(f"[tokenize] {split}: {n:,} examples", flush=True)
        # jobs: chunks of the (shuffled) order, each restricted to one file to make reads cheap
        chunk = args.chunk
        jobs = []
        for start in range(0, n, chunk):
            sub = rows[start:start + chunk]
            for f_idx in np.unique(sub[:, 0]):
                m = sub[:, 0] == f_idx
                jobs.append((start, m, str(files[f_idx]), sub[m, 1], args.max_len, args.head))
        seq_by_pos: dict[int, np.ndarray] = {}
        lengths = np.zeros(n, dtype=np.int32)
        names = np.empty(n, dtype=object)
        ttypes = np.empty(n, dtype=object)
        t0 = time.time()
        n_trunc = 0
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(_tok_job, j[2:]): j[:2] for j in jobs}
            done = 0
            for fut in as_completed(futs):
                start, m = futs[fut]
                seqs, nm, tt, nt = fut.result()
                n_trunc += nt
                positions = start + np.flatnonzero(m)
                for p, s, a, b in zip(positions, seqs, nm, tt):
                    seq_by_pos[int(p)] = s
                    lengths[p] = len(s); names[p] = a; ttypes[p] = b
                done += 1
                if done % 200 == 0:
                    print(f"[tokenize] {done}/{len(jobs)} jobs {time.time()-t0:.0f}s", flush=True)
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        tokens = np.lib.format.open_memmap(str(out_prefix) + ".tokens.npy", mode="w+", dtype=np.uint16, shape=(int(offsets[-1]),))
        for i in range(n):
            tokens[offsets[i]:offsets[i + 1]] = seq_by_pos.pop(i)
        tokens.flush(); del tokens
        np.save(str(out_prefix) + ".offsets.npy", offsets)
        pq.write_table(pa.table({"task_name": pa.array(names.tolist(), pa.string()).dictionary_encode(),
                                 "template_type": pa.array(ttypes.tolist(), pa.string()).dictionary_encode(),
                                 "length": pa.array(lengths.tolist(), pa.int32())}),
                       str(out_prefix) + ".meta.parquet", compression="zstd")
        meta = {"n": n, "total_tokens": int(offsets[-1]), "mean_length": float(lengths.mean()),
                "p50": int(np.percentile(lengths, 50)), "p95": int(np.percentile(lengths, 95)),
                "max": int(lengths.max()), "n_truncated": n_trunc, "max_len": args.max_len,
                "layout": "[CLS] inputs ' ' [unused0] target [SEP]; target is at offsets[i+1]-2",
                "sizes": sorted(args.sizes)}
        out_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[tokenize] {split} done: {meta}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["filter", "sample", "tokenize", "all"])
    ap.add_argument("--raw", default="data/flan_raw")
    ap.add_argument("--out", default="data/flan_atp")
    ap.add_argument("--sizes", default="2000000,5000000,20000000")
    ap.add_argument("--n-heldout", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--head", type=int, default=512, help="tokens kept from the start when truncating (rest from the end)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.sizes = [int(x) for x in args.sizes.split(",")]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = ["filter", "sample", "tokenize"] if args.stage == "all" else [args.stage]
    for st in stages:
        {"filter": stage_filter, "sample": stage_sample, "tokenize": stage_tokenize}[st](args)


if __name__ == "__main__":
    main()
