import csv
import os
from pathlib import Path


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.all_atom_parse import load_structure_data, index_to_token, restype_3to1


MODEL=os.environ.get("MODEL")  # Change to "best_ce" if you want to evaluate the best_ce model instead

INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/INFERENCE/negative_sequences (1)noX.csv")
PT_PATH = Path(f"/dapustor/nilufer/ADFLIP/results/benchmark/{MODEL}/test_set_109_adaptive_step_8_temp_0.1_noise_0.1_ther_0.9_argmax_1_ns_1_tfmr.pt")

PARSED_DIRS = [
    Path("/dapustor/nilufer/ADFLIP/dataset/test_set_109_parsed"),
    # Path("/dapustor/nilufer/ADFLIP/dataset/test_nucleotide_parsed"),
    # Path("/dapustor/nilufer/ADFLIP/dataset/test_small_molecule_parsed"),
]

OUT_DIR = Path(f"/dapustor/nilufer/ADFLIP/INFERENCE/Results_first_neg/{MODEL}")
SUMMARY_CSV = OUT_DIR / "summary.csv"
REPORT_CSV = OUT_DIR / "extraction_report.csv"
TOKEN_MAP_CSV = OUT_DIR / "logit_index_to_token.csv"
LOGPROB_DIR = OUT_DIR / "chain_log_prob_csvs"


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


def find_npz(pdb_id: str):
    for root in PARSED_DIRS:
        path = npz_path_for_pdb(root, pdb_id)
        if path.exists():
            return path
    return None


def get_chain_internal_id(structure, chain_label: str, center_chain_ids):
    if hasattr(structure, "asym_id_to_chain_index"):
        mapping = getattr(structure, "asym_id_to_chain_index")
        if isinstance(mapping, np.ndarray):
            if mapping.shape == ():
                mapping = mapping.item()
            elif len(mapping) == 1:
                mapping = mapping[0]
        if isinstance(mapping, dict) and chain_label in mapping:
            return int(mapping[chain_label])

    unique_chain_ids = np.unique(center_chain_ids)
    if len(unique_chain_ids) == 1:
        return int(unique_chain_ids[0])

    raise KeyError(f"Chain label {chain_label} not found in structure mapping")


def get_pred_sequence(entry):
    s_pred = entry.get("S_pred", "")
    if isinstance(s_pred, list):
        return s_pred[0] if s_pred else ""
    if isinstance(s_pred, str):
        return s_pred
    return ""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGPROB_DIR.mkdir(parents=True, exist_ok=True)

    pt_data = torch.load(PT_PATH, map_location="cpu")

    column_names = [token_label(index_to_token[i]) for i in range(len(index_to_token))]
    with open(TOKEN_MAP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["logit_column", "token", "column_name"])
        for i in range(len(index_to_token)):
            tok = index_to_token[i]
            writer.writerow([i, tok, token_label(tok)])

    summary_rows = []
    report_rows = []
    seen = set()

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            full_id = row.get("Sequence_ID", "").strip()
            gt_seq = row.get("GT_sequence", "").strip()

            if not full_id or "_" not in full_id:
                continue
            if full_id in seen:
                continue
            seen.add(full_id)

            pdb_id, chain_label = full_id.split("_", 1)
            pdb_id = pdb_id.lower()

            pt_key, entry = get_pt_entry(pt_data, pdb_id)
            if entry is None:
                report_rows.append({
                    "sequence_id": full_id,
                    "status": "missing_pt_entry",
                    "pt_key": "",
                    "npz_path": "",
                    "gt_len": len(gt_seq),
                    "extracted_len": "",
                    "match_true_vs_gt": "",
                    "match_pred_vs_gt": "",
                    "note": "",
                })
                continue

            if not isinstance(entry, dict):
                report_rows.append({
                    "sequence_id": full_id,
                    "status": "pt_entry_not_dict",
                    "pt_key": pt_key,
                    "npz_path": "",
                    "gt_len": len(gt_seq),
                    "extracted_len": "",
                    "match_true_vs_gt": "",
                    "match_pred_vs_gt": "",
                    "note": str(entry),
                })
                continue

            npz_path = find_npz(pdb_id)
            if npz_path is None:
                report_rows.append({
                    "sequence_id": full_id,
                    "status": "missing_npz",
                    "pt_key": pt_key,
                    "npz_path": "",
                    "gt_len": len(gt_seq),
                    "extracted_len": "",
                    "match_true_vs_gt": "",
                    "match_pred_vs_gt": "",
                    "note": "",
                })
                continue

            structure = load_structure_data(str(npz_path))

            center_mask = structure.is_center & structure.is_protein
            if hasattr(structure, "backbone_mask"):
                center_mask = center_mask & structure.backbone_mask

            center_chain_ids = np.asarray(structure.chain_id)[center_mask]

            used_whole_sequence = False
            try:
                chain_internal_id = get_chain_internal_id(structure, chain_label, center_chain_ids)
                chain_positions = np.where(center_chain_ids == chain_internal_id)[0]
                if len(chain_positions) == 0:
                    used_whole_sequence = True
                    chain_internal_id = ""
                    chain_positions = np.arange(len(center_chain_ids))
            except Exception:
                used_whole_sequence = True
                chain_internal_id = ""
                chain_positions = np.arange(len(center_chain_ids))

            s_true = entry.get("S_true", "")
            s_pred = get_pred_sequence(entry)
            logits = entry.get("ensemble_logits", None)

            if not isinstance(s_true, str) or logits is None:
                report_rows.append({
                    "sequence_id": full_id,
                    "status": "missing_required_keys",
                    "pt_key": pt_key,
                    "npz_path": str(npz_path),
                    "gt_len": len(gt_seq),
                    "extracted_len": "",
                    "match_true_vs_gt": "",
                    "match_pred_vs_gt": "",
                    "note": f"Available keys: {list(entry.keys())}",
                })
                continue

            logits = logits.cpu() if torch.is_tensor(logits) else torch.tensor(logits)

            if len(s_true) != logits.shape[0]:
                report_rows.append({
                    "sequence_id": full_id,
                    "status": "pt_len_mismatch",
                    "pt_key": pt_key,
                    "npz_path": str(npz_path),
                    "gt_len": len(gt_seq),
                    "extracted_len": "",
                    "match_true_vs_gt": "",
                    "match_pred_vs_gt": "",
                    "note": f"S_true len {len(s_true)} != logits rows {logits.shape[0]}",
                })
                continue

            if s_pred and len(s_pred) != len(s_true):
                report_rows.append({
                    "sequence_id": full_id,
                    "status": "pred_len_mismatch",
                    "pt_key": pt_key,
                    "npz_path": str(npz_path),
                    "gt_len": len(gt_seq),
                    "extracted_len": "",
                    "match_true_vs_gt": "",
                    "match_pred_vs_gt": "",
                    "note": f"S_pred len {len(s_pred)} != S_true len {len(s_true)}",
                })
                continue

            chain_seq_true = "".join(s_true[i] for i in chain_positions)
            chain_seq_pred = "".join(s_pred[i] for i in chain_positions) if s_pred else ""
            chain_logits = logits[chain_positions]
            chain_log_probs = F.log_softmax(chain_logits, dim=-1).numpy()

            perplexity_overall = entry.get("perplexity", "")
            if torch.is_tensor(perplexity_overall):
                perplexity_overall = float(perplexity_overall.item())

            match_true = chain_seq_true == gt_seq if gt_seq else ""
            match_pred = chain_seq_pred == gt_seq if gt_seq and chain_seq_pred else ""

            summary_rows.append({
                "sequence_id": full_id,
                "pdb_id": pdb_id,
                "chain_label": chain_label,
                "chain_internal_id": chain_internal_id,
                "used_whole_sequence": used_whole_sequence,
                "pt_key": pt_key,
                "npz_path": str(npz_path),
                "gt_sequence": gt_seq,
                "s_true_chain": chain_seq_true,
                "s_pred_chain": chain_seq_pred,
                "gt_len": len(gt_seq),
                "chain_len": len(chain_seq_true),
                "perplexity_overall": perplexity_overall,
                "match_true_vs_gt": match_true,
                "match_pred_vs_gt": match_pred,
            })

            safe_name = full_id.replace("/", "_")
            per_pos_rows = []
            for i in range(len(chain_positions)):
                row_out = {
                    "sequence_id": full_id,
                    "pdb_id": pdb_id,
                    "chain_label": chain_label,
                    "chain_internal_id": chain_internal_id,
                    "used_whole_sequence": used_whole_sequence,
                    "position_in_chain": i,
                    "aa_true": chain_seq_true[i],
                    "aa_pred": chain_seq_pred[i] if chain_seq_pred else "",
                }
                for cls_idx, col_name in enumerate(column_names):
                    row_out[col_name] = chain_log_probs[i, cls_idx]
                per_pos_rows.append(row_out)

            pd.DataFrame(per_pos_rows).to_csv(LOGPROB_DIR / f"{safe_name}.csv", index=False)

            report_rows.append({
                "sequence_id": full_id,
                "status": "ok",
                "pt_key": pt_key,
                "npz_path": str(npz_path),
                "gt_len": len(gt_seq),
                "extracted_len": len(chain_seq_true),
                "match_true_vs_gt": match_true,
                "match_pred_vs_gt": match_pred,
                "note": "" if not used_whole_sequence else f"Chain label {chain_label} not found; used whole sequence",
            })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "pdb_id",
                "chain_label",
                "chain_internal_id",
                "used_whole_sequence",
                "pt_key",
                "npz_path",
                "gt_sequence",
                "s_true_chain",
                "s_pred_chain",
                "gt_len",
                "chain_len",
                "perplexity_overall",
                "match_true_vs_gt",
                "match_pred_vs_gt",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "status",
                "pt_key",
                "npz_path",
                "gt_len",
                "extracted_len",
                "match_true_vs_gt",
                "match_pred_vs_gt",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Wrote summary CSV to: {SUMMARY_CSV}")
    print(f"Wrote extraction report to: {REPORT_CSV}")
    print(f"Wrote chain log-prob CSVs to: {LOGPROB_DIR}")
    print(f"Wrote token map to: {TOKEN_MAP_CSV}")


if __name__ == "__main__":
    main()
