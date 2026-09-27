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

**Full training run:** Completed 20,000 steps on the public [Kaggle kernel](https://www.kaggle.com/code/hexraei/notebookc1fbe077f9), v1. The measured numbers below come from the run's `runs/shoestring/training_report.json` in the published checkpoint bundle. The notebook's total runtime (8h 39m 15s) includes setup, export and attempted eval; it is not the training wall time. The benchmark stage failed on an export issue in that run, now fixed in the repo; no scores should be inferred from the training log.

| Metric | Measured value |
| --- | --- |
| Hardware | Tesla T4 (Kaggle GPU T4 x2 session) |
| Training wall time | 28,908.1 seconds (8h 1m 48s) |
| Training steps | 20,000 |
| Training tokens | 327,680,000 (batch 32 x sequence 512 x steps 20,000) |
| Estimated training FLOPs | 6.033e16 (6 x trainable params x training tokens) |
| Trainable parameters | 30,686,592 |
| Final training loss | 1.4755 |
| Final validation loss | 1.1959 (TinyStories validation) |

The FLOP estimate covers a standard 6-parameter-multiplier training approximation and is not a hardware profiler reading. Tokens are repeated training exposures, not a count of unique dataset tokens.

## Benchmark results (Track 01 eval set)

`bash scripts/run_eval.sh runs/shoestring/checkpoint.pt` runs all five. The training notebook did not produce eval scores because its HF export failed. The fix is committed; the scores below stay blank until the fixed suite runs against the real published checkpoint:

| Benchmark | Score |
| --- | --- |
| HellaSwag (acc) | _pending measured evaluation_ |
| ARC-Easy (acc) | _pending measured evaluation_ |
| PIQA (acc) | _pending measured evaluation_ |
| WinoGrande (acc) | _pending measured evaluation_ |
| WikiText-103 held-out slice (perplexity) | _pending measured evaluation_ |

This small TinyStories model may perform poorly on out-of-domain benchmarks. No above-random claim or score is made until evaluation completes. The interesting story is the measured efficiency per FLOP, with actual benchmark scores reported when available.

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
