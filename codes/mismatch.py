import csv
from pathlib import Path

import numpy as np

from data.all_atom_parse import load_structure_data


INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/finetuning_data/train_protein_mpnn_final_noX.csv")
PARSED_ROOT = Path("/dapustor/nilufer/ADFLIP/dataset/train_parsed")
OUTPUT_CSV = Path("/dapustor/nilufer/ADFLIP/finetuning_data/npz_mismatch_report.csv")


def pdb_to_npz(parsed_root: Path, pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    return parsed_root / pdb_id[1:3] / f"{pdb_id}.npz"


def to_sorted_unique_list(arr):
    return sorted(np.unique(arr).tolist())


def main():
    rows = []

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_id = row["Sequence_ID"].strip()
            gt_seq = row["GT_sequence"].strip()
            pdb_id = seq_id.split("_", 1)[0]

            npz_path = pdb_to_npz(PARSED_ROOT, pdb_id)
            if not npz_path.exists():
                rows.append(
                    {
                        "sequence_id": seq_id,
                        "gt_len": len(gt_seq),
                        "center_protein_len": "",
                        "center_protein_backbone_len": "",
                        "backbone_atom_count": "",
                        "backbone_residue_count": "",
                        "extra_backbone_residue_ids": "",
                        "missing_center_residue_ids": "",
                        "status": "missing_npz",
                    }
                )
                continue

            data = load_structure_data(str(npz_path))

            is_center = np.asarray(data.is_center).astype(bool)
            is_protein = np.asarray(data.is_protein).astype(bool)
            is_backbone = np.asarray(data.is_backbone).astype(bool)
            not_pad_mask = np.asarray(data.not_pad_mask).astype(bool)
            residue_index = np.asarray(data.residue_index)

            if hasattr(data, "backbone_mask"):
                backbone_mask = np.asarray(data.backbone_mask).astype(bool)
            else:
                backbone_mask = np.ones_like(is_backbone, dtype=bool)

            center_protein_mask = is_center & is_protein & not_pad_mask
            center_protein_backbone_mask = center_protein_mask & backbone_mask

            backbone_atom_mask = is_protein & is_backbone & not_pad_mask & backbone_mask

            center_res_ids = residue_index[center_protein_backbone_mask]
            backbone_res_ids = residue_index[backbone_atom_mask]

            center_res_set = set(to_sorted_unique_list(center_res_ids))
            backbone_res_set = set(to_sorted_unique_list(backbone_res_ids))

            extra_backbone = sorted(backbone_res_set - center_res_set)
            missing_center = sorted(center_res_set - backbone_res_set)

            rows.append(
                {
                    "sequence_id": seq_id,
                    "gt_len": len(gt_seq),
                    "center_protein_len": int(center_protein_mask.sum()),
                    "center_protein_backbone_len": int(center_protein_backbone_mask.sum()),
                    "backbone_atom_count": int(backbone_atom_mask.sum()),
                    "backbone_residue_count": len(backbone_res_set),
                    "extra_backbone_residue_ids": "|".join(map(str, extra_backbone)),
                    "missing_center_residue_ids": "|".join(map(str, missing_center)),
                    "status": "mismatch" if (extra_backbone or missing_center or int(backbone_atom_mask.sum()) != 4 * int(center_protein_backbone_mask.sum())) else "ok",
                }
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sequence_id",
            "gt_len",
            "center_protein_len",
            "center_protein_backbone_len",
            "backbone_atom_count",
            "backbone_residue_count",
            "extra_backbone_residue_ids",
            "missing_center_residue_ids",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote mismatch report to: {OUTPUT_CSV}")

    total = len(rows)
    mismatches = sum(r["status"] == "mismatch" for r in rows)
    oks = sum(r["status"] == "ok" for r in rows)
    missing = sum(r["status"] == "missing_npz" for r in rows)

    print(f"Total rows: {total}")
    print(f"OK rows: {oks}")
    print(f"Mismatch rows: {mismatches}")
    print(f"Missing NPZ rows: {missing}")


if __name__ == "__main__":
    main()
