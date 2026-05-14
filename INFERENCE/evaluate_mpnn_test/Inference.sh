#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-complex_lambda_1}"

export MODEL

echo "Running pipeline for MODEL=${MODEL}"

python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnn_test/extract_seq_logits.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnn_test/accuracy.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnn_test/SR_CRR.py
python /dapustor/nilufer/ADFLIP/INFERENCE/evaluate_mpnn_test/plot.py

echo "Pipeline complete for MODEL=${MODEL}"
