# #!/usr/bin/env bash
# set -euo pipefail

# export PYTHONPATH=$PWD:${PYTHONPATH:-}

# DATA_DIR="/dapustor/nilufer/ADFLIP/dataset"

# # 1) Download mmCIF files from RCSB (uncomment as needed)
# #    Train/valid: download asymmetric unit
# python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/train.json --out_dir ${DATA_DIR}/train
# python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/valid.json --out_dir ${DATA_DIR}/valid
# #    Test: download biological assembly 1 only
# python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/test_metal.json --out_dir ${DATA_DIR}/test_metal \
#   --base_url "https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
# python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/test_small_molecule.json --out_dir ${DATA_DIR}/test_small_molecule \
#   --base_url "https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
# python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/test_nucleotide.json --out_dir ${DATA_DIR}/test_nucleotide \
#   --base_url "https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"

# # 2) Parse mmCIF files into .npz (output: ${DATA_DIR}/*_parsed/)
# bash /dapustor/nilufer/ADFLIP/src/parser_dataset.sh

# # 3) Extract sequences from parsed data (output: FASTA + metadata CSV)
# bash /dapustor/nilufer/ADFLIP/src/extract_sequences_all.sh

# # 4) MMseqs2 clustering (requires mmseqs in PATH)
# bash /dapustor/nilufer/ADFLIP/src/cluster_train_valid_mmseqs.sh

# # 5) Build cluster split files (output: data/cluster/)
# python3 /dapustor/nilufer/ADFLIP/data/build_mmseqs_clusters.py \
#   --train_tsv /dapustor/nilufer/ADFLIP/data/cluster/mmseqs/mmseqs_train/train_clusters.tsv \
#   --valid_tsv /dapustor/nilufer/ADFLIP/data/cluster/mmseqs/mmseqs_valid/valid_clusters.tsv \
#   --out_dir /dapustor/nilufer/ADFLIP/data/cluster

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PWD:${PYTHONPATH:-}

DATA_DIR="/dapustor/nilufer/ADFLIP/dataset"
python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/test_metal.json --out_dir ${DATA_DIR}/test_metal 
python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/test_small_molecule.json --out_dir ${DATA_DIR}/test_small_molecule 
python3 /dapustor/nilufer/ADFLIP/data/download_data.py --json /dapustor/nilufer/ADFLIP/data/split/test_nucleotide.json --out_dir ${DATA_DIR}/test_nucleotide 
