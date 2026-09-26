# ShoestringLM

A 30.7M-parameter language model, written from scratch and trained from scratch, small enough to train on a free GPU and readable enough to grade.

Built for **Global Innovation Build Challenge V2, Track 01: Foundational LLM Development** (team Maverick).

---

## The idea, in plain words

Big language models cost millions to train. This project asks a smaller, more honest question: **how good a model can one student train from zero, with no pretrained weights, on free-tier hardware?** Every line of the model, the tokenizer, and the training loop is in this repo and fits in an afternoon of reading.

## What's inside

| Piece | File | What it does |
| --- | --- | --- |
| Model | `shoestring/model.py` | Decoder-only transformer: RMSNorm, rotary position embeddings, GELU MLP, tied input/output embeddings |
| Tokenizer | `shoestring/tokenizer.py` | Byte-level BPE, 8,192 tokens, trained on the training split itself |
| Training | `shoestring/train.py` | AdamW + cosine schedule, gradient clipping, val loss logging, automatic hardware/time/compute report |
| Param count | `shoestring/count_params.py` | Prints the full breakdown and checks it against the 50M cap |
| Eval | `scripts/run_eval.sh`, `scripts/eval_perplexity.py` | HellaSwag, ARC-Easy, PIQA, WinoGrande via lm-evaluation-harness + WikiText-103 perplexity |

## Parameter count (the 50M rule)

```
embeddings (tied with output head)        3,145,728
per transformer block                     1,377,024
all 20 blocks                            27,540,480
final norm                                    384
TOTAL trainable parameters               30,686,592   <- cap is 50,000,000: PASS
```

Reproduce with `python3 -m shoestring.count_params --config configs/shoestring_30m.json`. The output head **is** the embedding matrix (tied), so nothing is hidden from the count. Full config in `configs/shoestring_30m.json`.

## Rules compliance (short version)

- **From scratch:** weights are randomly initialized in `model.py`; no pretrained checkpoint, no fine-tuning, no distillation. The tokenizer is also trained from the same corpus.
- **Frameworks/datasets:** PyTorch and Hugging Face `tokenizers`/`datasets` are used (allowed), and the dataset is the public TinyStories corpus (allowed, credited below).
- **No hosted inference API:** the model is plain PyTorch in this repo.

## Compute report (required by Track 01)

The training script writes `training_report.json` on every run (hardware, wall time, tokens seen, FLOP estimate as `6 x params x tokens`).

**Smoke run (committed, reproducible):** 300 steps of the tiny config on a 2-core CPU, ~29k tokens/sec, loss 6.24 -> 4.22. Artifacts in `smoke_output/`.

**Full run:** 1 epoch of TinyStories (~550M tokens) at 30.7M params is ~1.0e17 FLOPs, which is roughly **5-8 hours on a single free Kaggle T4**. The exact measured numbers land here after the run:

| Metric | Value |
| --- | --- |
| Hardware | _filled from training_report.json after the Kaggle run_ |
| Total training time | _same_ |
| Tokens trained | _same_ |
| Estimated FLOPs | _same_ |

## Benchmark results (Track 01 eval set)

`bash scripts/run_eval.sh runs/shoestring/checkpoint.pt` runs all five. Numbers land after the full run:

| Benchmark | Score |
| --- | --- |
| HellaSwag (acc) | _after full run_ |
| ARC-Easy (acc) | _after full run_ |
| PIQA (acc) | _after full run_ |
| WinoGrande (acc) | _after full run_ |
| WikiText-103 held-out slice (perplexity) | _after full run_ |

Honest expectation, so judges can calibrate: a 30M model trained on TinyStories for one epoch will sit modestly above random on the four accuracy tasks (random is ~25%). The interesting story is the efficiency per FLOP, and we report exactly what we get.

## Run it

```bash
pip install -r requirements.txt

# prove the pipeline on your laptop in ~2 minutes (CPU):
bash scripts/run_smoke.sh

# the real run (GPU recommended; free Kaggle/Colab T4 works):
python3 -m shoestring.train --config configs/shoestring_30m.json \
  --steps 20000 --batch 32 --lr 3e-4 --out runs/shoestring

# generate from a checkpoint:
python3 -m shoestring.generate --checkpoint runs/shoestring/checkpoint.pt \
  --prompt "Once upon a time"
```

### Kaggle/Colab quick path

1. New notebook, enable GPU (T4).
2. `!git clone https://github.com/Hexraei/ShoestringLM && cd ShoestringLM && pip install -r requirements.txt`
3. Run the training cell above. Download `runs/shoestring/` when done.
4. `bash scripts/run_eval.sh runs/shoestring/checkpoint.pt` for the benchmark numbers.

## What was AI-assisted

Per the hackathon's AI policy: an AI coding assistant (Instinct, running Claude) wrote most of this code and this README with the author directing architecture, constraints, and review. The author owns the design decisions: model size and shape, tied embeddings, dataset choice, the shoestring compute budget, and all submission content.

## Credits

- **TinyStories** - Ronen Eldan, Yuanzhi Li (Microsoft Research, 2023). Public dataset of synthetic short stories.
- **lm-evaluation-harness** - EleutherAI, for the four benchmark tasks.
- **PyTorch**, **Hugging Face tokenizers + datasets** - frameworks and tooling (permitted by the rules).
- Model architecture follows standard published practice (RoPE from RoFormer, RMSNorm from Zhang & Sennrich, tied embeddings from Press & Wolf). Written from scratch in `model.py`.
