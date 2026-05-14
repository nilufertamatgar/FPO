#!/usr/bin/env bash

export PYTHONPATH=$PWD:${PYTHONPATH:-}

DATA_DIR="/dapustor/nilufer/ADFLIP/dataset"

# python3 /dapustor/nilufer/ADFLIP/data/parser_dataset.py --data_path ${DATA_DIR}/train/
# python3 /dapustor/nilufer/ADFLIP/data/parser_dataset.py --data_path ${DATA_DIR}/valid/
python3 /dapustor/nilufer/ADFLIP/data/parser_dataset.py --data_path ${DATA_DIR}/test_MPNN/
# python3 /dapustor/nilufer/ADFLIP/data/parser_dataset.py --data_path ${DATA_DIR}/test_small_molecule/
# python3 /dapustor/nilufer/ADFLIP/data/parser_dataset.py --data_path ${DATA_DIR}/test_nucleotide/
