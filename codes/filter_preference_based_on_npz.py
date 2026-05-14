import csv
from pathlib import Path

from data.all_atom_parse import load_structure_data


INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/finetuning_data/adflip_valid.csv")
PARSED_ROOT = Path("/dapustor/nilufer/ADFLIP/dataset/valid_parsed")
OUTPUT_CSV = Path("/dapustor/nilufer/ADFLIP/finetuning_data/adflip_valid_strict.csv")


def pdb_to_npz(parsed_root: Path, pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    return parsed_root / pdb_id[1:3] / f"{pdb_id}.npz"


def valid_length(npz_path: Path) -> int:
    data = load_structure_data(str(npz_path))
    mask = data.is_center & data.is_protein
    if hasattr(data, "backbone_mask"):
        mask = mask & data.backbone_mask
    return int(mask.sum())


def main():
    kept = 0
    skipped_missing_npz = 0
    skipped_pos_len = 0
    skipped_neg_len = 0

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                seq_id = row["Sequence_ID"].strip()
                pdb_id = seq_id.split("_", 1)[0]

                npz_path = pdb_to_npz(PARSED_ROOT, pdb_id)
                if not npz_path.exists():
                    skipped_missing_npz += 1
                    continue

                target_len = valid_length(npz_path)

                gt_seq = row["GT_sequence"].strip()
                if len(gt_seq) != target_len:
                    skipped_pos_len += 1
                    continue

                ok = True
                for k, v in row.items():
                    if k.startswith("Neg_seq_") and v and v.strip():
                        if len(v.strip()) != target_len:
                            ok = False
                            break

                if not ok:
                    skipped_neg_len += 1
                    continue

                writer.writerow(row)
                kept += 1

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Kept rows: {kept}")
    print(f"Skipped missing npz: {skipped_missing_npz}")
    print(f"Skipped positive length mismatch: {skipped_pos_len}")
    print(f"Skipped negative length mismatch: {skipped_neg_len}")


if __name__ == "__main__":
    main()
