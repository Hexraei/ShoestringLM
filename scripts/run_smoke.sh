#!/usr/bin/env bash
# End-to-end CPU smoke run: proves the whole pipeline in minutes on a laptop.
# Uses small slices of the TinyStories validation shard so nothing big downloads.
set -euo pipefail
python3 -m shoestring.train --config configs/smoke_tiny.json \
  --split 'validation[:3000]' --val-split 'validation[3000:3120]' \
  --steps 300 --batch 8 --lr 1e-3 --warmup 30 --val-every 60 \
  --out runs/smoke
python3 -m shoestring.count_params --config configs/shoestring_30m.json
python3 -m shoestring.generate --checkpoint runs/smoke/checkpoint.pt --max-new 60
