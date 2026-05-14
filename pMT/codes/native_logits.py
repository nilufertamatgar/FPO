import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from data.all_atom_parse import index_to_token, restype_3to1


PT_PATH = Path("/dapustor/nilufer/ADFLIP/results/benchmark/ADFLIP_v1/pMT_adaptive_step_8_temp_0.1_noise_0.1_ther_0.9_argmax_1_ns_1_tfmr.pt")
OUT_DIR = Path("/dapustor/nilufer/ADFLIP/pMT/extracted_log_probs_pMT")

SEQ_CSV = OUT_DIR / "pdb_sequences.csv"
TOKEN_MAP_CSV = OUT_DIR / "logit_index_to_token.csv"
ERROR_CSV = OUT_DIR / "error_entries.csv"
LOGPROBS_DIR = OUT_DIR / "log_prob_csvs"


def token_label(token):
    if token in restype_3to1:
        return f"AA_{restype_3to1[token]}"
    if token in {"A", "C", "G", "T", "U", "DA", "DC", "DG", "DT", "DU"}:
        return f"NT_{token}"
    return token


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGPROBS_DIR.mkdir(parents=True, exist_ok=True)

    data = torch.load(PT_PATH, map_location="cpu")

    seq_rows = []
    error_rows = []
    column_names = [token_label(index_to_token[i]) for i in range(len(index_to_token))]

    with open(TOKEN_MAP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["logit_column", "token", "column_name"])
        for i in range(len(index_to_token)):
            tok = index_to_token[i]
            writer.writerow([i, tok, token_label(tok)])

    for pdb_id, entry in data.items():
        if not isinstance(entry, dict):
            error_rows.append({
                "pdb_id": pdb_id,
                "entry_type": type(entry).__name__,
                "error_text": str(entry),
            })
            continue

        if "S_true" not in entry or "ensemble_logits" not in entry:
            error_rows.append({
                "pdb_id": pdb_id,
                "entry_type": type(entry).__name__,
                "error_text": f"Missing keys. Available keys: {list(entry.keys())}",
            })
            continue

        s_true = entry["S_true"]
        logits = entry["ensemble_logits"].cpu()
        log_probs = F.log_softmax(logits, dim=-1).numpy()

        seq_rows.append({
            "pdb_id": pdb_id,
            "sequence": s_true,
        })

        log_probs_df = pd.DataFrame(log_probs, columns=column_names)
        safe_name = pdb_id.replace("/", "_")
        log_probs_path = LOGPROBS_DIR / f"{safe_name}.csv"
        log_probs_df.to_csv(log_probs_path, index=False)

    with open(SEQ_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "sequence"])
        writer.writeheader()
        writer.writerows(seq_rows)

    with open(ERROR_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "entry_type", "error_text"])
        writer.writeheader()
        writer.writerows(error_rows)

    print(f"Wrote sequence CSV to: {SEQ_CSV}")
    print(f"Wrote token mapping CSV to: {TOKEN_MAP_CSV}")
    print(f"Wrote error-entry CSV to: {ERROR_CSV}")
    print(f"Wrote per-PDB log-prob CSVs to: {LOGPROBS_DIR}")
    print(f"Valid dict entries exported: {len(seq_rows)}")
    print(f"Skipped error/non-dict entries: {len(error_rows)}")


if __name__ == "__main__":
    main()
