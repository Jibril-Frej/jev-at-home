"""Ask the same questions about several states with every downloaded model, and compare the answers.

The cases file is JSON with a short name for each state and one `questions` object in the API format:

  {"states":    {"calm": "...", "furious": "..."},
   "questions": {"frustration": {"type": "score", "instructions": "...", "criteria": [...]}}}

For each model found in --models-dir, the script starts `jevhome serve`, sends one request per state
(all questions at once), then stops the server, so only one model is in memory at a time.
It prints one table per question (rows: states, columns: models) and saves the tables as Markdown.
A cell shows the noul value, the score, or the most likely choice with its probability.

Run (from the repository root, after `cargo build --release`):
  python3 scripts/compare_models.py scripts/compare_cases.json
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Released models, in table column order: (directory name, column label)
MODELS = [
    ("jevhome-ettin-1b", "Ettin-1B"), ("jevhome-L", "L"), ("jevhome-B", "B"), ("jevhome-E", "E"),
    ("jevhome-ettin-1b-int8", "Ettin-1B int8"), ("jevhome-L-int8", "L int8"),
    ("jevhome-B-int8", "B int8"), ("jevhome-E-int8", "E int8"),
]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"request failed ({e.code}): {e.read().decode()}")


def run_model(binary, model_dir, threads, states, questions):
    """Start a server for one model, ask every state, stop it. Returns {state name: answers}."""
    port = free_port()
    proc = subprocess.Popen([binary, "serve", str(model_dir), "--port", str(port), "--threads", str(threads)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):  # wait up to 60 s for the model to load
            if proc.poll() is not None:
                sys.exit(f"jevhome serve {model_dir} exited: {proc.stderr.read()}")
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=1)
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        else:
            sys.exit(f"jevhome serve {model_dir} did not start within 60 s")
        return {name: post(base + "/v1/systemone", {"state": state, "questions": questions})["answers"]
                for name, state in states.items()}
    finally:
        proc.terminate()
        proc.wait()


def cell(answer):
    if answer["type"] == "noul":
        return f"{answer['noul']:.2f}"
    if answer["type"] == "score":
        return f"{answer['score']:.2f}"
    return f"{answer['choice']} ({answer['probabilities'][answer['choice']]:.2f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", help="JSON file with `states` (name -> text) and `questions`")
    ap.add_argument("--models-dir", default="models", help="folder holding the downloaded models (default: models)")
    ap.add_argument("--binary", default="target/release/jevhome", help="the jevhome binary")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", help="Markdown file to write (default: <cases>.results.md)")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    states, questions = cases["states"], cases["questions"]
    models = [(d, label) for d, label in MODELS if (Path(args.models_dir) / d).is_dir()]
    if not models:
        sys.exit(f"no models found in {args.models_dir}")

    results = {}  # label -> {state name: answers}
    for d, label in models:
        print(f"running {label} ...", file=sys.stderr)
        results[label] = run_model(args.binary, Path(args.models_dir) / d, args.threads, states, questions)

    labels = [label for _, label in models]
    out = []
    for q, spec in questions.items():
        title = f"### {q} ({spec['type']})"
        if spec["type"] == "score" and isinstance(spec.get("criteria"), list):
            title += f": 0 = {spec['criteria'][0]}, {len(spec['criteria']) - 1} = {spec['criteria'][-1]}"
        rows = [f"| state | {' | '.join(labels)} |", f"|---|{'---|' * len(labels)}"]
        rows += [f"| {name} | {' | '.join(cell(results[l][name][q]) for l in labels)} |" for name in states]
        out.append("\n".join([title, "", *rows]))
    text = "\n\n".join(out) + "\n"

    print(text)
    out_path = Path(args.out or Path(args.cases).with_suffix(".results.md"))
    out_path.write_text(text)
    print(f"saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
