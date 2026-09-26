#!/usr/bin/env bash
# Benchmark eval per GIBC Track 01: HellaSwag, ARC-Easy, PIQA, WinoGrande
# via lm-evaluation-harness, plus WikiText-103 perplexity.
# Usage: bash scripts/run_eval.sh runs/shoestring/checkpoint.pt
set -euo pipefail
CKPT=${1:?checkpoint path required}
pip install -q lm-eval
python scripts/export_hf.py "$CKPT" runs/hf_export
lm_eval --model hf --model_args pretrained=runs/hf_export \
  --tasks hellaswag,arc_easy,piqa,winogrande \
  --batch_size 16 --output_path runs/eval_results.json
python scripts/eval_perplexity.py --checkpoint "$CKPT"
