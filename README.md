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

## Continued training on FineWeb-Edu (v2 and v3)

After the v1 TinyStories run, training continued from the same checkpoint on a public general-domain web sample (HuggingFaceFW/fineweb-edu, sample-10BT), same model, same tokenizer, lower LR (7e-5 cosine). The v3 run is public: [ShoestringLM v3 wider FineWeb training](https://www.kaggle.com/code/hexraei/shoestringlm-v3-wider-fineweb-training) (Kaggle, GPU T4 x2).

| Run | New data | Steps | Hardware | Wall time | Notebook |
| --- | --- | --- | --- | --- | --- |
| v1 | TinyStories | 20,000 | Tesla T4 (T4 x2 session) | 28,908.1 s | [public training](https://www.kaggle.com/code/hexraei/notebookc1fbe077f9) + [public eval](https://www.kaggle.com/code/hexraei/shoestringlm-gibc-track-01-eval-only/notebook) |
| v2 | 6,000 FineWeb-Edu docs | 4,000 + 4,000 | T4 x2 | recorded in run bundles | not published |
| v3 | 20,000 FineWeb-Edu docs (disjoint from v2's 6,000) | 8,000 (checkpoints at 4k/8k) | T4 x2 | 12,755.9 s | [public](https://www.kaggle.com/code/hexraei/shoestringlm-v3-wider-fineweb-training) |

The web sample has NOT been audited for overlap with benchmark questions, so v2/v3 scores are not contamination-free claims.

## Benchmark results (Track 01 eval set)

`bash scripts/run_eval.sh <checkpoint>` runs all five. Zero-shot unnormalized accuracy (`acc`), same eval code and same 200-row WikiText-103 validation slice (21,977 scored tokens) for every checkpoint. v1 scores come from the public eval-only notebook; v2 and v3 scores were measured with the same eval code in the Kaggle runs described above (v3 notebook linked; the v2 notebook is not published).

| Benchmark | v1 | v2 4k | v2 8k | v3 4k | v3 8k |
| --- | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | 0.2639 | 0.2671 | 0.2680 | 0.2664 | 0.2674 |
| ARC-Easy | 0.2626 | 0.2988 | 0.2971 | 0.3035 | 0.3009 |
| PIQA | 0.5620 | 0.5495 | 0.5490 | 0.5539 | 0.5544 |
| WinoGrande | 0.5114 | 0.5059 | 0.5107 | 0.5233 | 0.5130 |
| WikiText-103 PPL (200-row slice) | 628.71 | 33.47 | 34.45 | 28.06 | 25.51 |

Headline checkpoint: v3 at 4,000 steps. Versus v1: ARC-Easy +4.09 pts, WinoGrande +1.19 pts, HellaSwag +0.25 pts, PIQA -0.81 pts, perplexity 628.71 -> 28.06. v3 at 8k has the lowest perplexity (25.51) but gives back some ARC-Easy and WinoGrande. No checkpoint wins every metric. The multiple-choice scores remain near or below the task baselines on these out-of-domain benchmarks, and we make no competitive-performance claim. The v1 perplexity of 628.71 reflects a TinyStories-only model on Wikipedia text; the drop comes from adding general web text. The held-out slice is not the full WikiText-103 test set.

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
