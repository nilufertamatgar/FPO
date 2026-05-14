#!/usr/bin/env bash

CLUSTER_DIR="/dapustor/nilufer/ADFLIP/data/cluster/mmseqs"
TMP_DIR="/dapustor/nilufer/ADFLIP/tmp/mmseqs_tmp"

run_mmseqs() {
  local fasta="$1" prefix="$2"
  local out_dir="${CLUSTER_DIR}/mmseqs_${prefix}"
  local db="${out_dir}/${prefix}DB"
  local clu="${out_dir}/${prefix}CluDB"
  local tsv="${out_dir}/${prefix}_clusters.tsv"

  if [[ ! -f "${fasta}" ]]; then
    echo "[skip] ${fasta} not found"
    return
  fi

  mkdir -p "${out_dir}" "${TMP_DIR}_${prefix}"
  echo "[mmseqs] clustering ${prefix}"
  mmseqs createdb "${fasta}" "${db}"
  mmseqs linclust "${db}" "${clu}" "${TMP_DIR}_${prefix}" \
    --min-seq-id 0.3 -c 0.8 --cov-mode 1 --threads 32
  mmseqs createtsv "${db}" "${db}" "${clu}" "${tsv}"
}

run_mmseqs "${CLUSTER_DIR}/train_chains.fasta" "train"
run_mmseqs "${CLUSTER_DIR}/valid_chains.fasta" "valid"
