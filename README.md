# We have Jev at home

Open typed-decision models that run in real time on a plain CPU, with little memory.

- **Real time.** A decision takes 20–450 ms (median) depending on the model; the default model answers in about 65 ms (tested on my laptop's AMD Ryzen AI 9 HX PRO 370).
- **Low memory.** The models need 0.2 to 3.8 GB of RAM; the default one under 1 GB.
- **Plain CPU.** No GPU needed. All speed numbers here were measured on that laptop, with 4 threads.
- **Short inputs.** A decision reads at most 512 tokens; the models are made for short texts, not long documents.
- **Same API as Jev.** It serves the same `POST /v1/systemone` endpoint as Jev.

## Table of contents

<!-- toc -->

- [The models](#the-models)
- [Quick start](#quick-start)
- [API](#api)
- [Model details](#model-details)
  - [How the models compute probabilities](#how-the-models-compute-probabilities)
  - [Speed and memory (CPU, 4 threads)](#speed-and-memory-cpu-4-threads)
  - [Accuracy](#accuracy)
    - [The benchmarks](#the-benchmarks)
- [How the models were trained](#how-the-models-were-trained)
  - [Training data](#training-data)
- [Build](#build)
- [Compare the models](#compare-the-models)
- [Limitations](#limitations)
- [Licence](#licence)

<!-- tocstop -->

## The models

There are **8 models**: four sizes, each in a full-precision (fp32) and an int8 (quantised: smaller and faster,
slightly less accurate) version. They are listed largest first.

| Model | Size on disk (fp32 / int8) | Median latency (fp32 / int8) | Use it for |
|---|---|---|---|
| **Ettin-1B** | 3.9 GB / 1.0 GB | 445 / 159 ms | the best accuracy |
| **L** (large) | 1.5 GB / 393 MB | 182 / 73 ms | a middle ground |
| **B** (base) | 575 MB / 153 MB | 63 / 31 ms | the default: good accuracy and fast |
| **E** | 188 MB / 52 MB | 36 / 19 ms | the smallest footprint, least accurate |

Details, all benchmark results and how to choose are in [Model details](#model-details) below.

## Quick start

```bash
# 1. build the binary (needs Rust: https://rustup.rs, and a C compiler)
git clone https://github.com/Jibril-Frej/jev-at-home && cd jev-at-home
cargo build --release

# 2. download a model (here B, the default, 575 MB)
curl -LsSf https://hf.co/cli/install.sh | bash   # the Hugging Face `hf` CLI (standalone, no pip); or download the files from the model page
hf download jevhome/jevhome-B --local-dir models/jevhome-B --exclude 'pytorch/*'

# 3. load it once, then ask as often as you like
./target/release/jevhome serve models/jevhome-B            # --threads 4 --port 8009 --host 127.0.0.1 are the defaults
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
   "department":  {"type": "choice", "choice": "billing", "confidence": 0.73, "probabilities": {"billing": 0.82, "technical": 0.09, "sales": 0.09}},
   "escalate":    {"type": "noul", "noul": 0.54},
   "frustration": {"type": "score", "score": 1.74, "legend": {"0": "calm", "1": "slightly annoyed", "2": "frustrated", "3": "furious"},
                   "probabilities": {"0": 0.13, "1": 0.19, "2": 0.47, "3": 0.2}, "confidence": 0.78}},
 "usage": {"input_tokens": 143, "output_tokens": 0, "reasoning_tokens": 0},
 "latency_ms": 97.6}
```
This is a real response from jevhome-B on a laptop (AMD Ryzen AI 9 HX PRO 370, default 4 threads), reformatted for reading.
`output_tokens` is always 0: the models generate no text.

`latency_ms` is the server-side time for the whole request, covering all its questions.
`GET /v1/models` describes the loaded model.

To download all 8 models at once (`hf download` takes one repository at a time):

```bash
for m in ettin-1b L B E; do
  for v in "" -int8; do
    hf download jevhome/jevhome-$m$v --exclude 'pytorch/*' --local-dir models/jevhome-$m$v
  done
done
```

Remove `--exclude 'pytorch/*'` to also get the PyTorch weights of the fp32 models (only needed for further training).

## API

The API is the Jev one. A request has a `state` and a non-empty `questions` object;
`model` (echoed back) and `options` are optional.

| Question type | `criteria` | Answer |
|---|---|---|
| `noul` | optional `{"true": "...", "false": "..."}` descriptions | `noul`: probability that the statement is true |
| `choice` | object `{option: description or null}`, 1–255 options | `choice` (most probable option), `confidence`, `probabilities` |
| `score` | list of level descriptions, 1–255 levels | `score` (expected level, 0-based), `legend`, `probabilities`, `confidence` |

- **Confidence.** For `choice`, confidence is `(p_max - 1/k) / (1 - 1/k)`. For `score`, it is 1 minus the expected
  distance from the most probable level, normalised by `k - 1`. Probabilities are rounded to 2 decimals.
- **Errors.** An invalid request returns `422 {"detail": "..."}` with a message saying what is wrong; an unknown path returns `404`.
- **Several questions per request.** Each question is one forward pass. E reads the state separately from the questions (a bi-encoder, see
  Model details), so it encodes the state once per request, then only the questions.
- **One request at a time.** Requests are served in order on one model instance; for more throughput, run
  several servers.


`--threads N` sets how many CPU cores one decision uses (default 4, the setting of every number
below). More threads help mostly on long inputs; on a small VM use 1 or 2.

## Model details

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

- **Cross-encoders** (Ettin-1B, L, B) read the question, every option and the state together, in one sequence.
- **The bi-encoder** (E) reads the state and each option separately. It is the fastest and smallest model, but the
  least accurate. Both are explained under [How the models compute probabilities](#how-the-models-compute-probabilities).
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

### How the models compute probabilities

The backbones are standard text encoders (BERT-style). We keep each encoder as it is and add a small head on top.
The head gives one number (a *logit*) per option; the probabilities are a softmax over the options of the question,
with a temperature: `p = softmax(logits / T)`. A `noul` answer is the probability of the `true` option; a `score`
answer is the expected level, the sum of `level × probability`.

**Cross-encoders (Ettin-1B, L, B).** The question, the options and the state are one token sequence of at most
512 tokens, with a `[MASK]` token in front of each option:

```
[CLS] choice question: Which team should handle this ticket? [SEP]
      [MASK] billing: payments, invoices, refunds [MASK] technical: bugs and errors [MASK] sales: sales
      [SEP] I was charged twice for my subscription this month ... [SEP]
```

Each option is written as `id: description` (the id again when there is no description). A `noul` question has the
two options `true: The proposition is true.` and `false: The proposition is false.` (or your own descriptions), and a
`score` question one option per level: `0: calm`, `1: slightly annoyed`, ...

- The `[MASK]` tokens are only markers. The encoder's output vector at each marker summarises its option, read
  together with the question and the state.
- Changes to the base model: the masked-language-model output layer (which predicts a word at `[MASK]`) is not used
  for decisions. We add a **question-type vector** (one each for `choice`, `score` and `noul`), added to every
  output vector, and a **scorer**, a two-layer MLP (LayerNorm, Linear, GELU, Linear to 1 value) applied at each marker.
  Its output is the option's logit. This adds 0.6M (B) to 3.2M (Ettin-1B) parameters.
- The word-prediction layer is kept only during training, on the FLAN replay batches (see training).

**Bi-encoder (E).** The state and the options never share a sequence:

- The state is encoded alone (at most 512 tokens) into one vector `s` (the `[CLS]` output).
- Each option is encoded as the pair `"<type> question: <instructions>"` + option text (at most 160 tokens) into
  one vector `q`.
- A scorer MLP reads `[s, q, s × q, |s − q|]` and gives the option's logit. This adds 0.6M parameters; the encoder is unchanged.
- Because the state does not depend on the question, several questions about one state need only one state
  encoding. The price is accuracy on questions that need close reading of the state.

**Temperatures.** `T` is fitted after training on the calibration split of the dataset, one per question type and
number of options (2, 3–5, 6–10, 11+), by minimising the log-loss. They are stored in `calibration_u.json`, so
changing them changes the probabilities but never the chosen option.

### Speed and memory (CPU, 4 threads)

Latency p50 is the median time per decision, p90 the time that 90% of decisions stay under.
One decision at a time on the 100-item latency set: short to long states, all three question types, 94 tokens median.
Hardware: a laptop with an AMD Ryzen AI 9 HX PRO 370, 4 threads, no GPU; each model was run twice and the table
shows the mean of the two runs. Latency is end to end: tokenisation, model, probabilities.

| Model | Latency p50 | Latency p90 | Peak RAM | Load time |
|---|---|---|---|---|
| Ettin-1B | 445 ms | 1328 ms | 3.8 GB | 1.7 s |
| L | 182 ms | 537 ms | 2.1 GB | 1.2 s |
| B | 63 ms | 187 ms | 0.9 GB | 0.6 s |
| E | 36 ms | 83 ms | 0.3 GB | 0.2 s |
| Ettin-1B int8 | 159 ms | 532 ms | 1.2 GB | 0.8 s |
| L int8 | 73 ms | 212 ms | 0.5 GB | 0.6 s |
| B int8 | 31 ms | 89 ms | 0.3 GB | 0.4 s |
| E int8 | 19 ms | 45 ms | 0.2 GB | 0.2 s |

For reference, not measured by us:
- **Jev API:** 0.67 s per decision on JevBench, network included (published run).

### Accuracy

The first table gives **accuracy**: the percentage of questions where the model's most likely option is the right one
(for a `score` question, the right level). The second gives **calibration**: whether the probabilities can be trusted.
Each column is a benchmark, described [below the tables](#the-benchmarks). All numbers were produced by the `jevhome`
binary itself (one decision at a time, the same code path as `serve`). The fp32 models match our PyTorch evaluation on
every item (7,931 per model, probabilities within 3.1e-5).

**Models compared** (rows):
- **Jev 1.13:** the numbers published for it; we could not run it on the other benchmarks.
- **Qwen3.5-4B:** the large model our models learn from (the *teacher*, see [training](#how-the-models-were-trained)).
- **[Laya](https://huggingface.co/convaiinnovations/laya)** (by Convai): another family of open typed-decision models,
  in its three released versions.
- **Our models:** the four fp32 models, then their int8 versions.

**Accuracy (%, higher is better).** The number in each column header is the number of questions.
Ext mean is the mean of TD through VitaminC.

| Model | JevBench Easy (48) | Standard (72) | Hard (111) | TD (1660) | Auth (144) | Pert (108) | ANLI r3 (1000) | Banking77 (500) | CLINC150 (500) | SST-5 (500) | HelpSteer2 (268) | BoolQ* (1000) | VitaminC (1000) | **Ext mean** | Procgen (720) | Procgen v2 (300) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 (published) | 100 | 99 | 73 | 74† | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Qwen3.5-4B (our teacher) | 100 | 99 | 61 | 62 | 85 | 74 | 51 | 84 | 96 | 54 | 35 | 83 | 70 | 69.4 | 73 | 58 |
| Laya | 96 | 69 | 35 | 37 | 64 | 70 | 39 | 75 | 97 | 36 | 43 | 82 | 80 | 62.3 | 43 | 43 |
| Laya typed-decisions | 98 | 65 | 27 | 76 | 65 | 64 | 38 | 80 | 97 | 47 | 40 | 82 | 78 | 66.7 | 48 | 44 |
| Laya multilingual | 90 | 40 | 33 | 35 | 59 | 47 | 36 | 74 | 95 | 34 | 30 | 76 | 76 | 56.3 | 42 | 39 |
| **Ettin-1B** | 100 | 96 | 42 | 65 | 83 | 76 | 45 | 91 | 98 | 57 | 37 | 86 | 80 | **71.7** | 96 | 72 |
| **L** | 100 | 89 | 35 | 65 | 72 | 71 | 40 | 90 | 98 | 56 | 37 | 86 | 78 | **69.4** | 92 | 72 |
| **B** | 100 | 78 | 31 | 63 | 66 | 56 | 37 | 90 | 96 | 55 | 35 | 82 | 71 | **65.1** | 88 | 72 |
| **E** | 92 | 67 | 25 | 59 | 53 | 33 | 34 | 90 | 96 | 37 | 35 | 65 | 60 | **56.2** | 75 | 64 |
| Ettin-1B int8 | 98 | 93 | 42 | 65 | 80 | 71 | 41 | 90 | 97 | 56 | 35 | 86 | 78 | **70.0** | 94 | 75 |
| L int8 | 98 | 85 | 33 | 63 | 69 | 63 | 39 | 89 | 96 | 55 | 34 | 80 | 75 | **66.4** | 88 | 72 |
| B int8 | 98 | 75 | 32 | 60 | 69 | 44 | 38 | 86 | 93 | 47 | 32 | 74 | 68 | **61.1** | 79 | 71 |
| E int8 | 85 | 68 | 23 | 59 | 51 | 39 | 35 | 89 | 95 | 34 | 35 | 63 | 60 | **56.0** | 74 | 65 |

- **†Jev on TD:** this score comes from LangWatch on a 1,965-item version of the split; ours is a 1,660-item version,
  so the two are not directly comparable.
- **int8 rows:** evaluated once, on the same items, after the fp32 models were final; nothing was tuned on them.
  Compared with its fp32 model, int8 gives the same answer on 94% of items (Ettin-1B), 91% (E), 90% (L)
  and 83% (B).

**Calibration (ECE, lower is better).** ECE (expected calibration error) measures how far the probabilities are from the
observed accuracy: when a model says 80%, is it right about 80% of the time? We use top-label ECE (the probability of
the chosen option against its accuracy). Ext mean is the mean over TD through VitaminC, as above.
Jev is missing: only its accuracy is published.

| Model | JevBench Easy+Standard (120) | Hard (111) | TD (1660) | Auth (144) | Pert (108) | ANLI r3 (1000) | Banking77 (500) | CLINC150 (500) | SST-5 (500) | HelpSteer2 (268) | BoolQ* (1000) | VitaminC (1000) | **Ext mean** | Procgen (720) | Procgen v2 (300) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B (our teacher) | 0.035 | 0.116 | 0.118 | 0.068 | 0.085 | 0.291 | 0.061 | 0.010 | 0.075 | 0.273 | 0.051 | 0.140 | **0.117** | 0.096 | 0.042 |
| Laya | 0.083 | 0.192 | 0.168 | 0.110 | 0.111 | 0.429 | 0.152 | 0.023 | 0.291 | 0.053 | 0.091 | 0.066 | **0.149** | 0.252 | 0.281 |
| Laya typed-decisions | 0.198 | 0.171 | 0.214 | 0.131 | 0.111 | 0.325 | 0.043 | 0.008 | 0.054 | 0.057 | 0.048 | 0.055 | **0.105** | 0.116 | 0.176 |
| Laya multilingual | 0.241 | 0.394 | 0.309 | 0.164 | 0.246 | 0.506 | 0.169 | 0.026 | 0.265 | 0.168 | 0.156 | 0.145 | **0.216** | 0.368 | 0.405 |
| **Ettin-1B** | 0.088 | 0.224 | 0.049 | 0.060 | 0.093 | 0.298 | 0.038 | 0.010 | 0.126 | 0.229 | 0.062 | 0.035 | **0.100** | 0.030 | 0.044 |
| **L** | 0.083 | 0.257 | 0.026 | 0.091 | 0.107 | 0.301 | 0.031 | 0.015 | 0.113 | 0.208 | 0.048 | 0.038 | **0.098** | 0.038 | 0.065 |
| **B** | 0.084 | 0.288 | 0.043 | 0.064 | 0.108 | 0.329 | 0.019 | 0.036 | 0.149 | 0.208 | 0.033 | 0.048 | **0.104** | 0.054 | 0.063 |
| **E** | 0.103 | 0.246 | 0.037 | 0.089 | 0.250 | 0.206 | 0.034 | 0.023 | 0.052 | 0.137 | 0.027 | 0.050 | **0.090** | 0.073 | 0.064 |
| Ettin-1B int8 | 0.077 | 0.187 | 0.056 | 0.054 | 0.099 | 0.327 | 0.044 | 0.011 | 0.140 | 0.220 | 0.064 | 0.027 | **0.104** | 0.026 | 0.033 |
| L int8 | 0.104 | 0.220 | 0.045 | 0.081 | 0.083 | 0.264 | 0.056 | 0.042 | 0.171 | 0.144 | 0.034 | 0.027 | **0.095** | 0.037 | 0.041 |
| B int8 | 0.115 | 0.256 | 0.042 | 0.096 | 0.166 | 0.236 | 0.042 | 0.041 | 0.144 | 0.141 | 0.009 | 0.046 | **0.096** | 0.059 | 0.063 |
| E int8 | 0.072 | 0.286 | 0.036 | 0.127 | 0.214 | 0.249 | 0.032 | 0.021 | 0.046 | 0.195 | 0.015 | 0.065 | **0.100** | 0.070 | 0.068 |

#### The benchmarks

| Column | Benchmark | Questions asked | Type |
|---|---|---|---|
| JevBench Easy / Standard / Hard | [JevBench](https://benchmarkheaven.com), an independent public benchmark of Jev-style decisions, in three difficulty tiers | everyday decisions, up to multi-step reasoning over long states | all three |
| TD | test split of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | typed decisions written for Jev-like APIs | all three |
| Auth / Pert | the authored set and the perturbation set of [SemIf](https://github.com/TheoLeeCJ/SemIf) | weigh evidence, apply a rule, select a candidate | choice (3) |
| ANLI r3 | [ANLI](https://huggingface.co/datasets/facebook/anli), round 3 test | does the premise entail, contradict or not settle the hypothesis? (adversarial) | choice (3) |
| Banking77 | [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77) test | which intent is this bank-customer message? | choice (16 candidate intents) |
| CLINC150 | [CLINC150](https://huggingface.co/datasets/DeepPavlov/clinc150) test | which intent is this assistant request, or is it out of scope? | choice (16 candidate intents) |
| SST-5 | [SST-5](https://huggingface.co/datasets/SetFit/sst5) test | how positive is this movie-review sentence? | score (5 levels) |
| HelpSteer2 | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) validation | how helpful is this assistant response? | score (5 levels) |
| BoolQ* | [BoolQ](https://huggingface.co/datasets/google/boolq) validation | does the passage answer the question with yes? | noul |
| VitaminC | [VitaminC](https://huggingface.co/datasets/tals/vitaminc) test | does the evidence support, refute or not settle the claim? | choice (3) |
| Procgen / Procgen v2 | held-out items from our own procedural generators (see [training data](#training-data)), with states never seen in training; v2 focuses on multi-step, trap and time/number questions | in-distribution skill | noul, choice, score |

- **Ext mean** is the mean of TD through VitaminC. The Procgen columns are left out of it, because they come from the
  same generators as part of the training data.
- **No test data in training.** JevBench, TD and SemIf are filtered out of the training data, and the other columns
  use held-out test/validation splits. \*BoolQ is the exception: about 10% of its validation passages have
  near-duplicates in the training data, so treat that column as optimistic.

Honest summary:
- **Short, everyday decisions:** all our fp32 models except E get the easy tier fully right. Ettin-1B is close to Jev and
  to its own teacher on the standard tier.
- **Hard tier:** it stays far behind Jev (42 vs 73). Multi-step reasoning over long states is where a
  one-pass encoder loses to a large model.
- **int8:** Ettin-1B int8 loses little (1.7 points external, 2 standard-tier items). The smaller models lose more
  (L 3.1, B 4.0 external); E int8 is about as accurate as E, but drops 3 easy-tier items.
- **Calibration:** on the external benchmarks our models are at least as well calibrated as their teacher (Ext mean
  ECE 0.090–0.104 against 0.117). On the JevBench hard tier they are over-confident (0.19–0.29 against 0.12).

## How the models were trained

The four fp32 models follow **one protocol**: same data, same losses, same selection. Only the learning
rates and batch sizes differ per backbone. The training code, with the exact command for each model, is in
[`training/`](training/README.md).

1. **Instruction-tuned backbone (B only).** ModernBERT-base is first instruction-tuned with the recipe of
   [*It's All in The [MASK]*](https://arxiv.org/abs/2502.03793) (Clavié, Cooper, Warner 2025):
   - data: 20M FLAN 2022 examples with single-token answers, Yelp tasks removed;
   - objective: 80% answer-token prediction, 20% masked-language-modelling, 1 epoch.

   L starts from Answer.AI's ModernBERT-Large-Instruct, which was trained the same way by its authors.
   Ettin-1B and E start from their released checkpoints.
   The int8 models are quantised from the final fp32 models after training (no extra training).
2. **Targets.** Every training row has a gold label, and a teacher distribution from
   [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (bf16) read out in one pass over the option letters (as in SemIf).
   - The teacher reads each item twice, in the original and in reversed option order, and the two distributions are averaged.
   - A row is kept only if both orders agree on the answer, their total-variation distance is at most 0.2,
     and the teacher agrees with the gold label.
   - **Why a teacher and not only the gold label?** A gold label says only which answer is right, as if every
     case were certain, so a model trained on it alone learns to be over-confident. The teacher's probabilities also
     say how clear-cut each case is, so the model learns to give lower probabilities to ambiguous cases. That makes
     its probabilities more trustworthy (lower ECE, see [Accuracy](#accuracy)).
3. **Loss.** Cross-entropy on the gold label plus KL to the teacher distribution, with weight decay 0.1, 3 epochs and warm-up 5%.
   - **Distractor augmentation:** 30% of rows get irrelevant text added to the state, to teach the model to ignore it.
   - **Cross-encoders only:** option-order permutation of choice questions, and 20% replay of the FLAN
     instruction data to keep the backbone's general skills.
4. **Soups.** Three seeds per model are trained and uniformly weight-averaged ("model soup"; B's soup uses 2 of its 3 seeds).
   A soup is kept only if its dev accuracy is at most 1 point below the best seed; otherwise the best seed is used.
5. **Calibration.** One temperature per question type and number of options, fitted on a separate unfiltered dev set
   (`calibration_u.json` in each model folder).
6. **Export.** The models are exported to ONNX (fp32). The Python and Rust runtimes give the same answers, with probabilities equal to 1e-6.

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
the other platforms should work but are not tested yet, and the latencies above are for Linux on an AMD Ryzen laptop CPU.

`cargo build --release --no-default-features --features dynamic` instead loads `libonnxruntime.so`
from `ORT_DYLIB_PATH` at run time. That build was used to check that the built-in ONNX Runtime 1.28
gives the same token ids, answers and probabilities (to 1e-6) as the Python `onnxruntime` 1.26. With ONNX Runtime 1.26 loaded this way, decisions are 1–10% faster (3–8 ms less per decision, most visible on the small models).

Other subcommands: `jevhome probe` (decisions for a JSONL file) and `jevhome bench` (the latency protocol above).

## Compare the models

`scripts/compare_models.py` asks the same questions about several states with every model in `models/`
and prints one table per question: one row per state, one column per model.
A cell shows the `noul` value, the `score`, or the most likely `choice` with its probability.
Each model is loaded, asked every state, then stopped, so only one model is in memory at a time.
It needs all 8 models downloaded in `models/` (see [Quick start](#quick-start)); missing models are skipped.

```bash
python3 scripts/compare_models.py scripts/compare_cases.json   # also saves the tables to scripts/compare_cases.results.md
```

The cases file gives each state a short name, and one `questions` object in the API format:

```json
{"states":    {"calm": "Hi, I think I was charged twice ...", "furious": "UNACCEPTABLE. You charged me TWICE ..."},
 "questions": {"frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                               "criteria": ["calm", "slightly annoyed", "frustrated", "furious"]}}}
```

Results of [scripts/compare_cases.json](scripts/compare_cases.json) (the Quick start ticket, written at four levels of frustration):

**department** (choice)

| state | Ettin-1B | L | B | E | Ettin-1B int8 | L int8 | B int8 | E int8 |
|---|---|---|---|---|---|---|---|---|
| calm | billing (0.93) | billing (0.92) | billing (0.88) | billing (0.77) | billing (0.94) | billing (0.90) | billing (0.85) | billing (0.77) |
| slightly_annoyed | billing (0.89) | billing (0.93) | billing (0.82) | billing (0.63) | billing (0.79) | billing (0.93) | billing (0.93) | billing (0.56) |
| frustrated | billing (0.95) | billing (0.96) | billing (0.86) | billing (0.51) | billing (0.90) | billing (0.96) | billing (0.82) | billing (0.49) |
| furious | billing (0.98) | billing (0.95) | billing (0.97) | billing (0.64) | billing (0.96) | billing (0.98) | billing (0.95) | billing (0.73) |

**escalate** (noul)

| state | Ettin-1B | L | B | E | Ettin-1B int8 | L int8 | B int8 | E int8 |
|---|---|---|---|---|---|---|---|---|
| calm | 0.52 | 0.50 | 0.34 | 0.44 | 0.49 | 0.50 | 0.54 | 0.45 |
| slightly_annoyed | 0.68 | 0.56 | 0.54 | 0.45 | 0.68 | 0.38 | 0.59 | 0.42 |
| frustrated | 0.75 | 0.59 | 0.52 | 0.39 | 0.76 | 0.65 | 0.46 | 0.35 |
| furious | 0.80 | 0.80 | 0.49 | 0.43 | 0.77 | 0.61 | 0.57 | 0.43 |

**frustration** (score: 0 = calm, 3 = furious)

| state | Ettin-1B | L | B | E | Ettin-1B int8 | L int8 | B int8 | E int8 |
|---|---|---|---|---|---|---|---|---|
| calm | 1.28 | 1.26 | 1.08 | 1.49 | 1.29 | 1.15 | 1.31 | 1.57 |
| slightly_annoyed | 1.95 | 2.01 | 1.74 | 1.61 | 1.99 | 2.04 | 1.81 | 1.61 |
| frustrated | 2.45 | 2.41 | 2.14 | 1.75 | 2.38 | 2.10 | 2.17 | 1.88 |
| furious | 2.64 | 2.68 | 2.41 | 1.77 | 2.62 | 2.59 | 2.15 | 1.79 |

What this shows:

- All models send every version to billing; the larger ones are more confident.
- Ettin-1B, L and B put the four frustration levels in the right order, but the scores stay away from the ends
  (about 1.1–1.3 for calm, 2.4–2.7 for furious): the score is the probability-weighted average level, so
  spread-out probabilities pull it towards the middle. E barely separates the levels.
- `escalate` rises with frustration for Ettin-1B and L; B and E stay between about 0.35 and 0.6.

## Limitations

- **English only.**
- **Input length.** At most 512 tokens per decision; a longer state is cut. Not made for long texts.
- **Hard reasoning.** The models are far from Jev on the hard tier (multi-step reasoning, long states, tricky wording).
- **Calibration.** Probabilities are calibrated on our dev data; check them on your own data before thresholding.
- **Upstream data.** See the backbones' model cards for the data they were pre-trained on.

## Licence

- **Model weights and dataset:** CC BY-NC 4.0 (non-commercial). Several training sources are
  non-commercial or share-alike; see the dataset card.
- **Code:** MIT (see [LICENSE](LICENSE)).

This is an independent project. It is not affiliated with TypeSafe AI (Jev) or Convai (Laya).
