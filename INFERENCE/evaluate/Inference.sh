#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-beta0.2alpha1_lr0.001_ADFLIP_yyh_best}"

export MODEL

echo "Running pipeline for MODEL=${MODEL}"

python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate/extract_seq_logits.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate/accuracy.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate/SR_CRR.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate/plot.py

echo "Pipeline complete for MODEL=${MODEL}"
