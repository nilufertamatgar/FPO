import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.all_atom_parse import load_structure_data, index_to_token, restype_3to1


INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/Finetune_data_1/train_protein_mpnn_final_noX.csv")
PARSED_ROOT = Path("/dapustor/nilufer/ADFLIP/dataset/train_parsed")

REPORT_CSV = Path("/dapustor/nilufer/ADFLIP/finetuning_data/chain_npz_alignment_report.csv")
LOG_PATH = Path("/dapustor/nilufer/ADFLIP/finetuning_data/chain_npz_alignment_log.txt")


def pdb_to_npz(parsed_root: Path, pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    return parsed_root / pdb_id[1:3] / f"{pdb_id}.npz"


def extract_chain_sequence(structure, chain_internal_id):
    center_mask = np.asarray(structure.is_center).astype(bool)
    protein_mask = np.asarray(structure.is_protein).astype(bool)
    chain_mask = np.asarray(structure.chain_id) == chain_internal_id

    if hasattr(structure, "backbone_mask"):
        backbone_mask = np.asarray(structure.backbone_mask).astype(bool)
    else:
        backbone_mask = np.ones_like(center_mask, dtype=bool)

    seq_mask = center_mask & protein_mask & chain_mask & backbone_mask
    toks = np.asarray(structure.residue_token)[seq_mask]

    seq = []
    for tok in toks:
        token_name = index_to_token.get(int(tok), "<UNK>")
        seq.append(restype_3to1.get(token_name, "X"))
    return "".join(seq), int(seq_mask.sum())


def get_chain_internal_id(structure, chain_label: str):
    if hasattr(structure, "asym_id_to_chain_index"):
        mapping = getattr(structure, "asym_id_to_chain_index")
        if isinstance(mapping, dict) and chain_label in mapping:
            return int(mapping[chain_label])

    raise ValueError(f"Could not map chain label {chain_label} to internal chain_id")


def process_row(task):
    row_num, row = task
    seq_id = row.get("Sequence_ID", "").strip()
    gt_seq = row.get("GT_sequence", "").strip()

    base_result = {
        "row_num": row_num,
        "sequence_id": seq_id,
        "gt_len": len(gt_seq) if gt_seq else "",
        "gt_sequence": gt_seq,
        "npz_path": "",
        "chain_label": "",
        "chain_internal_id": "",
        "npz_chain_center_protein_backbone_len": "",
        "npz_chain_sequence": "",
        "match": "",
        "note": "",
    }

    if not seq_id or not gt_seq or "_" not in seq_id:
        result = dict(base_result)
        result["status"] = "malformed"
        return result, []

    pdb_id, chain_label = seq_id.split("_", 1)
    npz_path = pdb_to_npz(PARSED_ROOT, pdb_id)

    result = dict(base_result)
    result["chain_label"] = chain_label
    result["npz_path"] = str(npz_path)

    if not npz_path.exists():
        result["status"] = "missing_npz"
        return result, []

    try:
        structure = load_structure_data(str(npz_path))
        chain_internal_id = get_chain_internal_id(structure, chain_label)
        npz_chain_seq, npz_chain_len = extract_chain_sequence(structure, chain_internal_id)
        match = gt_seq == npz_chain_seq

        result.update(
            {
                "status": "ok" if match else "mismatch",
                "chain_internal_id": chain_internal_id,
                "npz_chain_center_protein_backbone_len": npz_chain_len,
                "npz_chain_sequence": npz_chain_seq,
                "match": match,
            }
        )

        log_lines = [
            f"Sequence_ID: {seq_id}",
            f"  GT len: {len(gt_seq)}",
            f"  NPZ chain len: {npz_chain_len}",
            f"  Chain label: {chain_label}",
            f"  Internal chain id: {chain_internal_id}",
            f"  GT sequence:  {gt_seq}",
            f"  NPZ sequence: {npz_chain_seq}",
            f"  Match: {match}",
            "-" * 100,
        ]
        return result, log_lines

    except Exception as exc:
        result["status"] = "error"
        result["note"] = repr(exc)
        log_lines = [
            f"Sequence_ID: {seq_id}",
            f"  ERROR: {repr(exc)}",
            "-" * 100,
        ]
        return result, log_lines


def load_rows(csv_path: Path):
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(enumerate(reader, start=2))


def main():
    tasks = load_rows(INPUT_CSV)
    rows = []
    log_blocks = []

    num_workers = max(1, mp.cpu_count() - 2)
    print(f"Using {num_workers} worker processes")

    with mp.Pool(processes=num_workers) as pool:
        for result, log_lines in tqdm(
            pool.imap_unordered(process_row, tasks),
            total=len(tasks),
            desc="Checking npz alignment",
        ):
            rows.append(result)
            if log_lines:
                log_blocks.append((result["row_num"], log_lines))

    rows.sort(key=lambda x: x["row_num"])
    log_blocks.sort(key=lambda x: x[0])

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_num",
                "sequence_id",
                "status",
                "gt_len",
                "gt_sequence",
                "npz_path",
                "chain_label",
                "chain_internal_id",
                "npz_chain_center_protein_backbone_len",
                "npz_chain_sequence",
                "match",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    flattened_logs = []
    for _, block in log_blocks:
        flattened_logs.extend(block)
    LOG_PATH.write_text("\n".join(flattened_logs), encoding="utf-8")

    print(f"Wrote report CSV to: {REPORT_CSV}")
    print(f"Wrote log file to: {LOG_PATH}")


if __name__ == "__main__":
    main()
