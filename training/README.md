# Training the jevhome models

This folder reproduces the four released fp32 models from the public dataset
[jevhome/jevhome-decisions](https://huggingface.co/datasets/jevhome/jevhome-decisions). It then exports them to the
ONNX layout that `jevhome` loads, and builds the int8 versions.

| model | backbone | kind |
|---|---|---|
| Ettin-1B | [jhu-clsp/ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b) | cross-encoder |
| L | [answerdotai/ModernBERT-Large-Instruct](https://huggingface.co/answerdotai/ModernBERT-Large-Instruct) | cross-encoder |
| B | [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), first instruction-tuned on FLAN (ATP, below) | cross-encoder |
| E | [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) | bi-encoder |

Every run below used **one NVIDIA A100 40GB**. The listed times are for that GPU. Training needs a GPU; export and
quantisation run on a CPU.

```bash
pip install -r requirements.txt
python from_hf.py            # -> data/jevhome/{train,dev,calibration}.jsonl (108,270 / 512 / 1,531 rows)
```

## What we changed in the base models

The backbones are unchanged: same layers, same weights to start from. Each model gets a small new decision head, and
the whole model is fine-tuned.

**Cross-encoders (Ettin-1B, L, B).** The state, the question and all its options go through the encoder together, as
one sequence. Each option starts with a `[MASK]` token, which serves as its marker:

```
[CLS] <type> question: <instructions> [SEP] [MASK] <id>: <option 1> [MASK] <id>: <option 2> ... [SEP] <state> [SEP]
```

The head reads the encoder's last hidden state at each option's `[MASK]` and adds a learned embedding of the question
type (`noul`, `choice`, `score`). A small scorer (LayerNorm, Linear, GELU, Linear to 1) then turns it into **one logit
per option**. The probabilities are the softmax of these logits over the question's options. The masked-LM
vocabulary head is not used for decisions. During training it only serves the FLAN replay batches, and the export
drops it.

**Bi-encoder (E).** The state and the options are encoded separately, and each text is represented by its `[CLS]`
vector:

- `s` = encoder(state), computed once per state;
- `q` = encoder("`<type> question: <instructions>`", "`<id>: <option>`"), one short text pair per option.

The **score** of an option is a small MLP (LayerNorm, Linear, GELU, Linear to 1) on `[s, q, s*q, |s-q|]`. The
probabilities are the softmax of the scores over the question's options. The state encoding does not depend on the
question, so several questions about one state cost one state pass plus the short option passes.

**Option order.** Every question type has a fixed option order: `noul` = [`true`, `false`], `choice` = the order of
its criteria, `score` = levels `0 .. k-1`. Each option text is `"<id>: <description>"`.

**Calibration.** Each model ships a `calibration_u.json` with one temperature `T` per question type and option-count
bucket (2, 3-5, 6-10, 11+). The temperatures are fitted by minimising the NLL on the dataset's `calibration` split.
jevhome divides the logits by `T` before the softmax.

## Training recipe (all four models)

- **Data.** The `train` split. Each row has a gold answer and/or the probabilities from a teacher model.
- **Losses.** `ce` is cross-entropy on the gold option. `kl` is the KL divergence to the teacher distribution.
- **Schedule.** 3 epochs, AdamW, linear warmup (5%) then linear decay, gradient clipping at 1.0 and weight decay
  0.1. The new head uses a higher learning rate (`--lr-head`).
- **Augmentation.**
  - From the second epoch on, the options of `choice` questions are shuffled (`--perm-aug choice`).
  - In each epoch, 30% of the rows get an unrelated sentence from another source added to their state; the label does
    not change (`--distract-frac 0.3`).
- **FLAN replay (cross-encoders only).** 20% of the steps are FLAN answer-token-prediction batches through the
  masked-LM head, used as a regulariser (`--replay`).
- **Seeds and soup.** Three seeds (0, 1, 2) are trained and their weights averaged ("model soup"). A seed is left out
  when its dev accuracy is more than 3 points below the best seed. For B, seed 2 diverged (0.75 dev accuracy vs 0.92
  and 0.93), so the released B averages seeds 0 and 1. If the soup ends up more than 1 point below the best seed, the
  best seed is used instead.
- **Temperatures.** They are fitted on the soup: `make_soup.sh` does the selection, the soup, the check and the fit.

### FLAN data (for B's backbone and for the replay)

```bash
hf download Open-Orca/FLAN --repo-type dataset --local-dir data/flan_raw          # FLAN 2022, 596 parquet files
python prepare_flan.py all --raw data/flan_raw --out data/flan_atp --workers 48    # single-token answers, 20M pool
python filter_flan.py data/flan_atp data/flan_atp_ny                              # drop the Yelp tasks -> 19.5M examples
```

The FLAN pool follows [It's All in The [MASK]](https://arxiv.org/abs/2502.03793) (Clavié et al. 2025):

- it keeps the examples whose answer is a single token;
- it caps the number of examples per task;
- it holds out MMLU/BBH;
- each example is laid out as `[CLS] input [unused0] answer [SEP]`.

The Yelp tasks are removed because the Yelp dataset's terms do not allow training models that are redistributed.

### B's backbone: ModernBERT-base + Answer Token Prediction (~12 h)

This step instruction-tunes ModernBERT-base with the paper's objective. Per example:

- with probability 0.8, the answer token is masked and predicted;
- otherwise, 30% of the tokens are masked and labelled with the `[MASK]` id itself ("dummy MLM").

The run makes one pass over the first 20M examples (50,754 steps).

```bash
python train_atp.py --data data/flan_atp_ny --n-examples 20000000 --model answerdotai/ModernBERT-base \
  --out checkpoints/modernbert-base-atp-20m-ny --tokens-per-batch 65536 --grad-accum 2 --lr 5e-5 \
  --warmup-frac 0.05 --atp-frac 0.8 --mlm-prob 0.3 --eval-every 2000 --save-every 5000 --attn sdpa
```

### Per-model commands

`D=data/jevhome`, `COMMON="--train $D/train.jsonl --dev $D/dev.jsonl --calib-dev $D/calibration.jsonl --loss ce,kl --weight-decay 0.1 --distract-frac 0.3"`

```bash
# Ettin-1B: 7,290 steps, ~1.3 h per seed
for s in 0 1 2; do python train_decision.py $COMMON --perm-aug choice --replay data/flan_atp_ny \
  --model jhu-clsp/ettin-encoder-1b --tokens-per-batch 8192 --grad-ckpt --lr 3e-5 --lr-head 5e-4 \
  --seed $s --out checkpoints/ET1B-s$s; done
./make_soup.sh checkpoints/SOUP-ET1B $D/calibration.jsonl checkpoints/ET1B-s{0,1,2}

# L: 3,586 steps, ~27 min per seed
for s in 0 1 2; do python train_decision.py $COMMON --perm-aug choice --replay data/flan_atp_ny \
  --model answerdotai/ModernBERT-Large-Instruct --tokens-per-batch 16384 --lr 5e-5 --lr-head 5e-4 \
  --seed $s --out checkpoints/L-s$s; done
./make_soup.sh checkpoints/SOUP-L $D/calibration.jsonl checkpoints/L-s{0,1,2}

# B: 1,807 steps, ~13 min per seed (needs the ATP backbone above)
for s in 0 1 2; do python train_decision.py $COMMON --perm-aug choice --replay data/flan_atp_ny \
  --model checkpoints/modernbert-base-atp-20m-ny --tokens-per-batch 32768 --lr 8e-5 --lr-head 5e-4 \
  --seed $s --out checkpoints/B-s$s; done
./make_soup.sh checkpoints/SOUP-B $D/calibration.jsonl checkpoints/B-s{0,1,2}

# E: 5,026 steps, ~12.5 min per seed
for s in 0 1 2; do python train_embed.py $COMMON \
  --model ibm-granite/granite-embedding-small-english-r2 --budget 20000 --lr 1e-4 --lr-head 5e-4 \
  --seed $s --out checkpoints/E-s$s; done
./make_soup.sh checkpoints/SOUP-E $D/calibration.jsonl checkpoints/E-s{0,1,2}
```

Each run writes `train_log.jsonl` (loss, accuracy, and dev metrics every 500 steps) and `train_summary.json` (final
dev accuracy, NLL, Brier and ECE, per source and per question type).

## Export and int8

```bash
python export_onnx.py checkpoints/SOUP-B models/B          # fp32: model.onnx (E: encoder.onnx + scorer.onnx)
python quantize.py models/B models/B-int8                  # int8 ("dyn8-e8")
./jevhome serve models/B                                   # or: jevhome probe models/B items.jsonl out.jsonl
```

**`export_onnx.py`** writes the graphs plus the tokenizer, `config.json`, `decision_config.json` and
`calibration_u.json`. In these graphs, the last encoder layer is computed only at the positions the head reads (the
`[MASK]` markers, or E's `[CLS]`). This is faster and gives the same answers. Graphs over 2 GB (Ettin-1B) come with a
`model.onnx.data` file.

**`quantize.py`** quantises two parts of the model:

- every linear layer: int8 per-channel weights, with the activations quantised at run time (ONNX Runtime
  `quantize_dynamic`);
- the vocabulary embedding table: int8 with one scale per row.

It needs no calibration data.

## Files

| file | what |
|---|---|
| `from_hf.py` | dataset parquet -> training records |
| `prepare_flan.py`, `filter_flan.py` | FLAN answer-token-prediction pool, Yelp tasks removed |
| `train_atp.py` | ATP instruction tuning of ModernBERT-base (B's backbone) |
| `train_decision.py` | cross-encoder training (Ettin-1B, L, B) |
| `train_embed.py` | bi-encoder training (E) |
| `soup.py`, `soup_check.py`, `fit_temperature.py`, `make_soup.sh` | seed averaging, fallback check, temperatures |
| `export_onnx.py`, `quantize.py` | fp32 ONNX export and int8 quantisation for jevhome |
| `jevtrain/` | shared code: option rendering, heads, losses, batching, evaluation |

## Reproducibility

The recipe is exactly the one used for the released models. The results will still differ slightly:

- GPU kernels are not bit-deterministic;
- the teacher probabilities are stored in the dataset as float32;
- `Open-Orca/FLAN` may change between revisions.

As B's seed 2 shows, a seed can occasionally diverge. The soup rule above drops such a seed.

## Licence

This code is MIT, like the rest of the repository. The dataset and the trained weights are CC BY-NC 4.0.
