# We have Jev at home

Open, Jev-like typed-decision models that run on a plain CPU.

You give a model a **state** (any text or JSON) and one or more **questions**. Each question is a
yes/no (`noul`), a pick-one (`choice`) or a graded level (`score`). The model answers every question
with **calibrated probabilities** in one forward pass: no generation and no reasoning tokens.

- **`jevhome`** is a single 33 MB binary with ONNX Runtime built in: no Python and no GPU; one `cargo build` makes it.
- **Same API as Jev and Jeeves.** It serves the same `POST /v1/systemone` endpoint as Jev and
  [PostHog's Jeeves](https://github.com/PostHog/jeeves), with the same request and response format,
  so existing clients work unchanged.
- **Fast on 4 CPU cores.** A decision takes about 35–70 ms (median, E and B) on 4 cores of a generic server CPU.

## Quick start

```bash
# 1. build the binary (needs Rust: https://rustup.rs, and a C compiler)
git clone https://github.com/Jibril-Frej/jev-at-home && cd jev-at-home
cargo build --release && cp target/release/jevhome .

# 2. a model (here B, 575 MB)
pip install -U huggingface_hub   # or download the files by hand from the model page
hf download jevhome/jevhome-B --local-dir models/jevhome-B --exclude 'pytorch/*'

# 3. load it once, then ask as often as you like
./jevhome serve models/jevhome-B            # --threads 4 --port 8009 --host 127.0.0.1 are the defaults
```

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my subscription this month and nobody answers my emails.",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this ticket?",
                    "criteria": {"billing": "payments, invoices, refunds", "technical": "bugs and errors", "sales": null}},
    "escalate":    {"type": "noul", "instructions": "Should this be escalated to a human manager?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["calm", "slightly annoyed", "frustrated", "furious"]}
  }
}'
```

```json
{"model": "jevhome-B",
 "answers": {
   "department":  {"type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {"billing": 0.93, "technical": 0.05, "sales": 0.02}},
   "escalate":    {"type": "noul", "noul": 0.61},
   "frustration": {"type": "score", "score": 2.1, "legend": {"0": "calm", "1": "slightly annoyed", "2": "frustrated", "3": "furious"},
                   "probabilities": {"0": 0.03, "1": 0.17, "2": 0.45, "3": 0.35}, "confidence": 0.74}},
 "usage": {"input_tokens": 211, "output_tokens": 160, "reasoning_tokens": 0},
 "latency_ms": 95.3}
```
<!-- TODO: replace this example output with a real response from the released B model. -->

`latency_ms` is the server-side time for the whole request, covering all its questions.
`GET /v1/models` describes the loaded model.

## API

The API is the Jev / Jeeves one. A request has a `state` and a non-empty `questions` object;
`model` (echoed back) and `options` are optional.

| Question type | `criteria` | Answer |
|---|---|---|
| `noul` | optional `{"true": "...", "false": "..."}` descriptions | `noul`: probability that the statement is true |
| `choice` | object `{option: description or null}`, 1–255 options | `choice` (most probable option), `confidence`, `probabilities` |
| `score` | list of level descriptions, 1–255 levels | `score` (expected level, 0-based), `legend`, `probabilities`, `confidence` |

- **Confidence.** For `choice`, confidence is `(p_max - 1/k) / (1 - 1/k)`. For `score`, it is 1 minus the expected
  distance from the most probable level, normalised by `k - 1`. Probabilities are rounded to 2 decimals, as in
  Jeeves.
- **Errors.** An invalid request returns `422 {"detail": "..."}`, with the same checks and messages as Jeeves; an unknown path returns `404`.
- **Reasoning options.** The Jeeves options `think`, `max_think`, `nothink_threshold` and `return_reasoning` are
  accepted and ignored: these models never generate reasoning tokens.
- **Several questions per request.** Each question is one forward pass. The bi-encoder (E) encodes the state once
  per request, then only the questions.
- **One request at a time.** Requests are served in order on one model instance; for more throughput, run
  several servers.
- **Existing clients.** The Jeeves Python SDK (`jeeves_sdk`) talks to `127.0.0.1:8009` by default, so it works
  unchanged against `jevhome serve`.

`--threads N` sets how many CPU cores one decision uses (default 4, the setting of every number
below). More threads help mostly on long inputs; on a small VM use 1 or 2.

## Models

| Model | Download | Backbone | Params | Architecture | Disk |
|---|---|---|---|---|---|
| **Ettin-1B** | [jevhome/jevhome-ettin-1b](https://huggingface.co/jevhome/jevhome-ettin-1b) | [jhu-clsp/ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b) | ~1.0B | cross-encoder | 3.9 GB |
| **L** | [jevhome/jevhome-L](https://huggingface.co/jevhome/jevhome-L) | [answerdotai/ModernBERT-Large-Instruct](https://huggingface.co/answerdotai/ModernBERT-Large-Instruct) | ~396M | cross-encoder | 1.5 GB |
| **B** | [jevhome/jevhome-B](https://huggingface.co/jevhome/jevhome-B) | ModernBERT-base + our FLAN instruction tuning (see below) | ~150M | cross-encoder | 575 MB |
| **E** | [jevhome/jevhome-E](https://huggingface.co/jevhome/jevhome-E) | [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) | ~48M | bi-encoder | 188 MB |
| **Ettin-1B int8** | [jevhome/jevhome-ettin-1b-int8](https://huggingface.co/jevhome/jevhome-ettin-1b-int8) | Ettin-1B, int8 | ~1.0B | cross-encoder | 1.0 GB |
| **L int8** | [jevhome/jevhome-L-int8](https://huggingface.co/jevhome/jevhome-L-int8) | L, int8 | ~396M | cross-encoder | 393 MB |
| **B int8** | [jevhome/jevhome-B-int8](https://huggingface.co/jevhome/jevhome-B-int8) | B, int8 | ~150M | cross-encoder | 153 MB |
| **E int8** | [jevhome/jevhome-E-int8](https://huggingface.co/jevhome/jevhome-E-int8) | E, int8 | ~48M | bi-encoder | 52 MB |

- **Cross-encoders** (Ettin-1B, L, B) read the question, every option and the state in one sequence of up to 512 tokens.
  A small head scores the option markers.
- **The bi-encoder** (E) encodes the state (up to 512 tokens) and each question+option (up to 160 tokens)
  separately, and a small MLP scores each pair. It is the fastest and smallest model, but the least accurate.
- **Parameter counts** are estimated from the fp32 weight files.
- **Disk** is the model folder (ONNX weights, tokenizer, configs) in MiB/GiB.
- **int8 models** are the same networks with dynamic int8 quantisation (ONNX Runtime `quantize_dynamic`: int8
  weights, activations quantised on the fly per call) and an int8 token-embedding table. They are 1.7-2.7x
  faster (more for the larger models) and about 4x smaller, but they are *different models*: they change the
  answer on 6-17% of items and lose up to 4 points on the external mean (see Accuracy).

Which model to pick:
- **B** is the default: good accuracy and fast.
- **Ettin-1B** is for the best accuracy when about 0.5 s per decision is fine.
- **Ettin-1B int8** is the best accuracy per millisecond: faster than L and more accurate.
- **B int8** is for speed: as fast as E and clearly more accurate.
- **E** / **E int8** are for the smallest footprint (under 200 MB of RAM for the int8 version).

### Speed and memory (CPU, 4 threads)

One decision at a time on the 100-item latency set: short to long states, all three question types, 94 tokens median.
Hardware: 4 cores of an AMD EPYC 9654, no GPU. Latency is end to end: tokenisation, model, probabilities.
<!-- Source: results/cpu-jevhome-ab, jobs 16045+16046 (both run orders averaged); int8: results/cpu-jevhome-int8, job 16067 (2 runs averaged). ORT 1.26 (dynamic build) is 1-10% faster: see Build. -->

| Model | Latency p50 | Latency p90 | Peak RAM | Load time |
|---|---|---|---|---|
| Ettin-1B | 447 ms | 1321 ms | 3.9 GB | 4.5 s |
| L | 194 ms | 560 ms | 2.1 GB | 4.3 s |
| B | 69 ms | 197 ms | 0.9 GB | 1.6 s |
| E | 37 ms | 88 ms | 0.3 GB | 0.6 s |
| Ettin-1B int8 | 168 ms | 506 ms | 1.2 GB | 1.6 s |
| L int8 | 80 ms | 225 ms | 0.5 GB | 1.0 s |
| B int8 | 36 ms | 102 ms | 0.3 GB | 0.7 s |
| E int8 | 22 ms | 50 ms | 0.2 GB | 0.3 s |

For reference, not measured by us:
- **Jev API:** 0.67 s per decision on JevBench, network included (published run).

### Accuracy

All numbers were produced by the `jevhome` binary itself (one decision at a time, the same code path as `serve`).
The fp32 models match our PyTorch evaluation on every item (7,931 per model, probabilities within 3.1e-5).
No test set was used for training or model selection:
- **JevBench:** never trained on (training data is filtered against it).
- **External columns:** held-out test/validation splits.

| Model | JevBench Easy (48) | Standard (72) | Hard (111) | ECE easy+std | TD (1660) | Auth (144) | Pert (108) | ANLI r3 (1000) | Banking77 (500) | CLINC150 (500) | SST-5 (500) | HelpSteer2 (268) | BoolQ* (1000) | VitaminC (1000) | **Ext mean** | Procgen (720) | Procgen v2 (300) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 (published) | 100 | 99 | 73 | n/a | 74† | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Qwen3.5-4B (our teacher) | 100 | 99 | 61 | 0.035 | 62 | 85 | 74 | 51 | 84 | 96 | 54 | 35 | 83 | 70 | 69.4 | 73 | 58 |
| Laya | 96 | 69 | 35 | 0.083 | 37 | 64 | 70 | 39 | 75 | 97 | 36 | 43 | 82 | 80 | 62.3 | 43 | 43 |
| Laya typed-decisions | 98 | 65 | 27 | 0.198 | 76 | 65 | 64 | 38 | 80 | 97 | 47 | 40 | 82 | 78 | 66.7 | 48 | 44 |
| Laya multilingual | 90 | 40 | 33 | 0.241 | 35 | 59 | 47 | 36 | 74 | 95 | 34 | 30 | 76 | 76 | 56.3 | 42 | 39 |
| **Ettin-1B** | 100 | 96 | 42 | 0.088 | 65 | 83 | 76 | 45 | 91 | 98 | 57 | 37 | 86 | 80 | **71.7** | 96 | 72 |
| **L** | 100 | 89 | 35 | 0.083 | 65 | 72 | 71 | 40 | 90 | 98 | 56 | 37 | 86 | 78 | **69.4** | 92 | 72 |
| **B** | 100 | 78 | 31 | 0.084 | 63 | 66 | 56 | 37 | 90 | 96 | 55 | 35 | 82 | 71 | **65.1** | 88 | 72 |
| **E** | 92 | 67 | 25 | 0.103 | 59 | 53 | 33 | 34 | 90 | 96 | 37 | 35 | 65 | 60 | **56.2** | 75 | 64 |
| Ettin-1B int8 | 98 | 93 | 42 | 0.077 | 65 | 80 | 71 | 41 | 90 | 97 | 56 | 35 | 86 | 78 | **70.0** | 94 | 75 |
| L int8 | 98 | 85 | 33 | 0.104 | 63 | 69 | 63 | 39 | 89 | 96 | 55 | 34 | 80 | 75 | **66.4** | 88 | 72 |
| B int8 | 98 | 75 | 32 | 0.115 | 60 | 69 | 44 | 38 | 86 | 93 | 47 | 32 | 74 | 68 | **61.1** | 79 | 71 |
| E int8 | 85 | 68 | 23 | 0.072 | 59 | 51 | 39 | 35 | 89 | 95 | 34 | 35 | 63 | 60 | **56.0** | 74 | 65 |

- **JevBench:** accuracy (%) on the public items. ECE is the top-label calibration error on the easy and standard tiers.
- **Other columns:**
  - TD = typed-decisions test (LocalLLaMA/typed-decisions).
  - Auth / Pert = the SemIf authored and perturbation sets.
  - The rest are standard datasets turned into typed decisions.
  - Ext mean is the mean of TD through VitaminC.
- **Procgen columns** are held-out states from our procedural generators, and are *not* in Ext mean. They show in-distribution skill.
- **Jev:** only its published JevBench numbers exist. †: Jev's typed-decisions score comes from LangWatch on a
  1,965-item version of the split; ours is a 1,660-item version, so the two are not directly comparable.
- **int8 rows:** evaluated once, on the same items, after the fp32 models were final; nothing was tuned on them.
  Compared with its fp32 model, int8 gives the same answer on 94% of items (Ettin-1B), 91% (E), 90% (L)
  and 83% (B).
- **\*BoolQ:** about 10% of BoolQ validation passages have near-duplicates in the training data, so treat it as
  optimistic.

Honest summary:
- **Short, everyday decisions:** all our fp32 models except E get the easy tier fully right. Ettin-1B is close to Jev and
  to its own teacher on the standard tier.
- **Hard tier:** it stays far behind Jev (42 vs 73). Multi-step reasoning over long states is where a
  one-pass encoder loses to a large model.
- **int8:** Ettin-1B int8 loses little (1.7 points external, 2 standard-tier items). The smaller models lose more
  (L 3.1, B 4.0 external); E int8 is about as accurate as E, but drops 3 easy-tier items.

## How the models were trained

All four models follow **one protocol**: same data, same losses, same selection. Only the learning
rates and batch sizes differ per backbone.

1. **Instruction-tuned backbone (B only).** ModernBERT-base is first instruction-tuned with the recipe of
   [*It's All in The [MASK]*](https://arxiv.org/abs/2502.03793) (Clavié, Cooper, Warner 2025):
   - data: 20M FLAN 2022 examples with single-token answers, Yelp tasks removed;
   - objective: 80% answer-token prediction, 20% masked-language-modelling, 1 epoch.

   L starts from Answer.AI's ModernBERT-Large-Instruct, which was trained the same way by its authors.
   Ettin-1B and E start from their released checkpoints.
2. **Targets.** Every training row has a gold label, and a teacher distribution from
   [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (bf16) read out in one pass over the option letters (SemIf-style).
   - The teacher reads each item twice, in the original and in reversed option order, and the two distributions are averaged.
   - A row is kept only if both orders agree on the answer, their total-variation distance is at most 0.2,
     and the teacher agrees with the gold label.
3. **Loss.** Cross-entropy on the gold label plus KL to the teacher distribution, with weight decay 0.1, 3 epochs and warm-up 5%.
   - **Distractor augmentation:** 30% of rows get irrelevant text added to the state, to teach the model to ignore it.
   - **Cross-encoders only:** option-order permutation of choice questions, and 20% replay of the FLAN
     instruction data to keep the backbone's general skills.
4. **Soups.** Three seeds per model are trained and uniformly weight-averaged ("model soup"; B's soup uses 2 of its 3 seeds).
   A soup is kept only if its dev accuracy is at most 1 point below the best seed; otherwise the best seed is used.
5. **Calibration.** One temperature per question type and number of options, fitted on a separate unfiltered dev set
   (`calibration_u.json` in each model folder).
6. **Not released.** A larger bi-encoder (from gte-modernbert-base, 149M parameters) was trained the same way.
   B is both faster and more accurate, so it is not released.
7. **Export.** The models are exported to ONNX (fp32). The Python and Rust runtimes give the same answers, with probabilities equal to 1e-6.

### Training data

The mixture has **108,270 training rows** plus a 512-row dev set. It is published as
[jevhome/jevhome-decisions](https://huggingface.co/datasets/jevhome/jevhome-decisions).

| Part | Rows | Share | What it is |
|---|---|---|---|
| Public datasets (mostly human-labelled) | 52,624 | 49% | 37 datasets recast as typed decisions, including NLI (MultiNLI, WANLI, ANLI, VitaminC, FEVER, SciTail, TemporalNLI, FOL-NLI, bAbI-NLI), QA and reading (BoolQ, SQuAD v2, PubMedQA, MC-TACO, LogiQA, ReClor, RuleTaker), intents (CLINC150, Banking77, HWU64, MASSIVE, customer support tickets), topic and sentiment (AG News, SST-5, GoEmotions, STS-B), response quality and judging (HelpSteer2/3, Prometheus, UltraFeedback, JudgeLM, Math-Shepherd, unanswerable math), safety (BeaverTails, Aegis, Civil Comments), tool calls (Glaive function calling), and LocalLLaMA/typed-decisions |
| Procedural | 46,019 | 43% | 20 program-generated families where the answer is computed, not guessed: table lookup, date and time, unit thresholds, numeric bins, probability, state tracking, truth chains, rule chains, trade-offs, policy rules / abstention, tool-schema checks, answer and format adequacy, intent traps, and Reasoning Gym puzzles (knights and knaves, family relations, calendar, coin flips, isomorphism) |
| LLM-synthetic | 9,627 | 9% | Short typed decisions written by Qwen3.5-4B from rubric specs (facts, policies, adequacy, quality / severity / urgency / tone levels), plus contrastive twins with one decisive fact flipped |

Data hygiene:
- **Contamination filter.** Every row is checked against JevBench and the SemIf sets: rows with 8-gram / embedding overlap are dropped,
  and generated rows (synthetic and procedural) go through a stricter 5-gram gate. The test splits used above are never in the mixture.
- **Removed sources.** Yelp, Skywork-Reward and ContractNLI were removed for licence reasons, including from B's FLAN data.
  L's upstream ModernBERT-Large-Instruct was trained by Answer.AI on a FLAN sample that includes Yelp.
- **Dataset licences** are listed per source in the dataset card.

## Build

```bash
cargo build --release      # downloads a prebuilt ONNX Runtime (1.28, static) at build time
./target/release/jevhome serve <model_dir>
```

`cargo install --git https://github.com/Jibril-Frej/jev-at-home` installs `jevhome` into `~/.cargo/bin` instead.
The build fetches ONNX Runtime from the [`ort`](https://ort.pyke.io) crate's prebuilt binaries, which exist for Linux
(x86-64, ARM64), macOS (Apple Silicon) and Windows (x86-64). It is tested on Linux x86-64, which also covers WSL;
the other platforms should work but are not tested yet, and the latencies above are for Linux on an AMD EPYC.

`cargo build --release --no-default-features --features dynamic` instead loads `libonnxruntime.so`
from `ORT_DYLIB_PATH` at run time. That build was used to check that the built-in ONNX Runtime 1.28
gives the same token ids, answers and probabilities (to 1e-6) as the Python `onnxruntime` 1.26. With ONNX Runtime 1.26 loaded this way, decisions are 1–10% faster (3–8 ms less per decision, most visible on the small models).

Other subcommands: `jevhome probe` (decisions for a JSONL file) and `jevhome bench` (the latency protocol above).

## Limitations

- **English only.**
- **Input length.** States longer than 512 tokens are truncated.
- **Hard reasoning.** The models are far from Jev on the hard tier (multi-step reasoning, long states, tricky wording).
- **Calibration.** Probabilities are calibrated on our dev data; check them on your own data before thresholding.
- **Upstream data.** See the backbones' model cards for the data they were pre-trained on.

## Licence

- **Model weights and dataset:** CC BY-NC 4.0 (non-commercial). Several training sources are
  non-commercial or share-alike; see the dataset card.
- **Code:** MIT (see [LICENSE](LICENSE)).

This is an independent project. It is not affiliated with TypeSafe AI (Jev), PostHog (Jeeves) or Convai (Laya).
