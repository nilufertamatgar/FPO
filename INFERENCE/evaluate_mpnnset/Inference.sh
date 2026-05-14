#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-complex_lambda_1}"

export MODEL

echo "Running pipeline for MODEL=${MODEL}"

python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnnset/extract_seq_logits.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnnset/accuracy.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnnset/SR_CRR.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnnset/plot.py

echo "Pipeline complete for MODEL=${MODEL}"
