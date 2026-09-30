"""int8 version of an exported model (export_onnx.py output): the "dyn8-e8" scheme of the released int8 models.

  dynamic int8  every linear layer (MatMul with a constant weight) gets int8 per-channel weights, its activations are
                quantised at run time (MatMulInteger); attention score products and everything else stay fp32. The
                scorer's Gemm layers are first rewritten as MatMul + Add (exact), so they are quantised too.
  -e8           the vocabulary embedding table is stored as int8 with one fp32 scale per row (Gather, Cast, Mul).
No calibration data is needed. The graphs are portable; ONNX Runtime optimises them when loading.
usage: quantize.py models/<name> models/<name>-int8
"""
import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from export_onnx import FILES, consolidate

BIG = 1 << 30  # graphs above 1 GiB are written with external data


def save(model, path):
    stale = path.parent / (path.name + ".data")
    if stale.exists():  # onnx appends external data to an existing file
        stale.unlink()
    size = sum(t.ByteSize() for t in model.graph.initializer)
    if size > BIG:
        onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=path.name + ".data", size_threshold=1024)
    else:
        onnx.save_model(model, str(path))
    return size > BIG


def gemm_to_matmul(model):
    """Gemm(A, B, C) with alpha = beta = 1 and a constant B -> MatMul(A, B') + C (exact rewrite)."""
    from onnx import helper, numpy_helper
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes = []
    for n in g.node:
        if n.op_type != "Gemm" or n.input[1] not in inits:
            nodes.append(n)
            continue
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        assert at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0 and not at.get("transA", 0), n.name
        w = numpy_helper.to_array(inits[n.input[1]])
        if at.get("transB", 0):
            w = w.T
        wn = n.input[1] + "_mm"
        g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w), wn))
        if len(n.input) > 2 and n.input[2]:
            mid = n.output[0] + "_mm"
            nodes.append(helper.make_node("MatMul", [n.input[0], wn], [mid], name=n.name + "_mm"))
            nodes.append(helper.make_node("Add", [mid, n.input[2]], [n.output[0]], name=n.name + "_bias"))
        else:
            nodes.append(helper.make_node("MatMul", [n.input[0], wn], [n.output[0]], name=n.name + "_mm"))
    del g.node[:]
    g.node.extend(nodes)
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    return model


def embed_int8(model):
    """Vocabulary table (the Gather on a [vocab, hidden] initializer) -> int8 table + fp32 per-row scale."""
    from onnx import TensorProto, helper, numpy_helper
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    n = next(n for n in g.node if n.op_type == "Gather" and n.input[0] in inits and inits[n.input[0]].dims[0] > 50000)
    w = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
    scale = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-12) / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    base = n.input[0]
    g.initializer.remove(inits[base])
    g.initializer.extend([numpy_helper.from_array(q, base + "_i8"), numpy_helper.from_array(scale.astype(np.float32), base + "_scale")])
    ids, out = n.input[1], n.output[0]
    new = [helper.make_node("Gather", [base + "_i8", ids], [out + "_i8"], name=n.name + "_i8"),
           helper.make_node("Cast", [out + "_i8"], [out + "_f"], to=TensorProto.FLOAT, name=n.name + "_cast"),
           helper.make_node("Gather", [base + "_scale", ids], [out + "_s"], name=n.name + "_scale"),
           helper.make_node("Mul", [out + "_f", out + "_s"], [out], name=n.name + "_dq")]
    i = list(g.node).index(n)
    g.node.remove(n)
    for j, m in enumerate(new):
        g.node.insert(i + j, m)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="fp32 model directory (export_onnx.py output)")
    ap.add_argument("out")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = ["model.onnx"] if (src / "model.onnx").exists() else ["encoder.onnx", "scorer.onnx"]
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        (tmp / "prep").mkdir()
        (tmp / "q").mkdir()
        for f in files:
            m = gemm_to_matmul(onnx.load(str(src / f)))
            if f != "scorer.onnx":
                m = embed_int8(m)
            big = save(m, tmp / "prep" / f)
            quantize_dynamic(tmp / "prep" / f, tmp / "q" / f, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                             per_channel=True, use_external_data_format=big, extra_options={"MatMulConstBOnly": True})
        consolidate(tmp / "q", out)
    for f in FILES:
        shutil.copy2(src / f, out / f)
    mb = sum(p.stat().st_size for p in out.iterdir() if not p.name.endswith(".json")) / 2 ** 20
    print(f"{src} -> {out}: {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
