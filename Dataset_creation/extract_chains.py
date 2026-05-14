import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.all_atom_parse import load_structure_data, index_to_token, restype_3to1


FASTA_PATH = Path("/dapustor/nilufer/ADFLIP/Dataset_creation/SIFT_fasta.csv")
PT_PATH = Path("/dapustor/nilufer/ADFLIP/results/benchmark/ADFLIP_v1/metal_adaptive_step_8_temp_0.1_noise_0.1_ther_0.9_argmax_1_ns_1_tfmr.pt")
PARSED_ROOT = Path("/dapustor/nilufer/ADFLIP/dataset/test_metal_parsed")

OUT_DIR = Path("/dapustor/nilufer/ADFLIP/Dataset_creation/metal_extracted")
SEQ_CSV = OUT_DIR / "chain_sequences.csv"
REPORT_CSV = OUT_DIR / "extraction_report.csv"
LOGPROB_DIR = OUT_DIR / "chain_log_prob_csvs"
TOKEN_MAP_CSV = OUT_DIR / "logit_index_to_token.csv"


def read_fasta_like(path: Path):
    records = {}
    current_id = None
    current_seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        records[current_id] = "".join(current_seq)
    return records


def token_label(token):
    if token in restype_3to1:
        return f"AA_{restype_3to1[token]}"
    if token in {"A", "C", "G", "T", "U", "DA", "DC", "DG", "DT", "DU"}:
        return f"NT_{token}"
    return token


def possible_pt_keys(pdb_id: str):
    return [
        f"{pdb_id}.cif.gz",
        f"{pdb_id}.cif",
        f"{pdb_id}.pdb",
        pdb_id,
    ]


def get_pt_entry(pt_data, pdb_id: str):
    for key in possible_pt_keys(pdb_id):
        if key in pt_data:
            return key, pt_data[key]
    return None, None


def npz_path_for_pdb(parsed_root: Path, pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    return parsed_root / pdb_id[1:3] / f"{pdb_id}.npz"


def get_chain_internal_id(structure, chain_label: str, center_chain_ids):
    if hasattr(structure, "asym_id_to_chain_index"):
        mapping = getattr(structure, "asym_id_to_chain_index")
        if isinstance(mapping, dict) and chain_label in mapping:
            return int(mapping[chain_label])

    unique_chain_ids = np.unique(center_chain_ids)
    if len(unique_chain_ids) == 1:
        return int(unique_chain_ids[0])

    raise ValueError(f"Could not map chain label {chain_label} to internal chain_id")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGPROB_DIR.mkdir(parents=True, exist_ok=True)

    sift_records = read_fasta_like(FASTA_PATH)
    pt_data = torch.load(PT_PATH, map_location="cpu")

    column_names = [token_label(index_to_token[i]) for i in range(len(index_to_token))]
    with open(TOKEN_MAP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["logit_column", "token", "column_name"])
        for i in range(len(index_to_token)):
            tok = index_to_token[i]
            writer.writerow([i, tok, token_label(tok)])

    seq_rows = []
    report_rows = []

    for full_id, fasta_seq in sift_records.items():
        if "_" not in full_id:
            report_rows.append({
                "chain_id": full_id,
                "status": "malformed_id",
                "pt_key": "",
                "npz_path": "",
                "sift_len": len(fasta_seq),
                "extracted_len": "",
                "match": "",
                "note": "Expected format pdb_chain",
            })
            continue

        pdb_id, chain_label = full_id.split("_", 1)
        pt_key, entry = get_pt_entry(pt_data, pdb_id)

        if entry is None:
            report_rows.append({
                "chain_id": full_id,
                "status": "missing_pt_entry",
                "pt_key": "",
                "npz_path": "",
                "sift_len": len(fasta_seq),
                "extracted_len": "",
                "match": "",
                "note": "",
            })
            continue

        if not isinstance(entry, dict):
            report_rows.append({
                "chain_id": full_id,
                "status": "pt_entry_not_dict",
                "pt_key": pt_key,
                "npz_path": "",
                "sift_len": len(fasta_seq),
                "extracted_len": "",
                "match": "",
                "note": str(entry),
            })
            continue

        npz_path = npz_path_for_pdb(PARSED_ROOT, pdb_id)
        if not npz_path.exists():
            report_rows.append({
                "chain_id": full_id,
                "status": "missing_npz",
                "pt_key": pt_key,
                "npz_path": str(npz_path),
                "sift_len": len(fasta_seq),
                "extracted_len": "",
                "match": "",
                "note": "",
            })
            continue

        structure = load_structure_data(str(npz_path))

        center_mask = structure.is_center & structure.is_protein
        if hasattr(structure, "backbone_mask"):
            center_mask = center_mask & structure.backbone_mask

        center_chain_ids = np.asarray(structure.chain_id)[center_mask]

        try:
            chain_internal_id = get_chain_internal_id(structure, chain_label, center_chain_ids)
        except Exception as e:
            report_rows.append({
                "chain_id": full_id,
                "status": "chain_map_failed",
                "pt_key": pt_key,
                "npz_path": str(npz_path),
                "sift_len": len(fasta_seq),
                "extracted_len": "",
                "match": "",
                "note": repr(e),
            })
            continue

        chain_positions = np.where(center_chain_ids == chain_internal_id)[0]

        s_true = entry["S_true"]
        logits = entry["ensemble_logits"].cpu()

        if len(s_true) != logits.shape[0]:
            report_rows.append({
                "chain_id": full_id,
                "status": "pt_len_mismatch",
                "pt_key": pt_key,
                "npz_path": str(npz_path),
                "sift_len": len(fasta_seq),
                "extracted_len": "",
                "match": "",
                "note": f"S_true len {len(s_true)} != logits rows {logits.shape[0]}",
            })
            continue

        chain_seq = "".join(s_true[i] for i in chain_positions)
        chain_log_probs = F.log_softmax(logits, dim=-1).numpy()[chain_positions]

        match = chain_seq == fasta_seq

        seq_rows.append({
            "chain_id": full_id,
            "pdb_id": pdb_id,
            "chain_label": chain_label,
            "sequence": chain_seq,
        })

        safe_name = full_id.replace("/", "_")
        pd.DataFrame(chain_log_probs, columns=column_names).to_csv(
            LOGPROB_DIR / f"{safe_name}.csv", index=False
        )

        report_rows.append({
            "chain_id": full_id,
            "status": "ok",
            "pt_key": pt_key,
            "npz_path": str(npz_path),
            "sift_len": len(fasta_seq),
            "extracted_len": len(chain_seq),
            "match": match,
            "note": "" if match else "Extracted chain sequence does not match SIFT fasta sequence",
        })

    with open(SEQ_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chain_id", "pdb_id", "chain_label", "sequence"])
        writer.writeheader()
        writer.writerows(seq_rows)

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chain_id",
                "status",
                "pt_key",
                "npz_path",
                "sift_len",
                "extracted_len",
                "match",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Wrote chain sequence CSV to: {SEQ_CSV}")
    print(f"Wrote extraction report to: {REPORT_CSV}")
    print(f"Wrote chain log-prob CSVs to: {LOGPROB_DIR}")
    print(f"Wrote token map to: {TOKEN_MAP_CSV}")


if __name__ == "__main__":
    main()
