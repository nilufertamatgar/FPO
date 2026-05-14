import csv
import shutil
from pathlib import Path


CSV_PATH = Path("/dapustor/nilufer/ADFLIP/Finetune_data_1/ADFLIP_valid.csv")
SRC_ROOT = Path("/dapustor/nilufer/ADFLIP/dataset/valid_parsed")
DST_ROOT = Path("/dapustor/nilufer/ADFLIP/dataset/valid_parsed_subset")


def pdb_to_npz(parsed_root: Path, pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    return parsed_root / pdb_id[1:3] / f"{pdb_id}.npz"


def main():
    pdb_ids = set()

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("Sequence_ID", "").strip()
            if not seq_id or "_" not in seq_id:
                continue
            pdb_id = seq_id.split("_", 1)[0].lower()
            pdb_ids.add(pdb_id)

    copied = 0
    missing = []

    for pdb_id in sorted(pdb_ids):
        src = pdb_to_npz(SRC_ROOT, pdb_id)
        dst = pdb_to_npz(DST_ROOT, pdb_id)

        if not src.exists():
            missing.append(str(src))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Unique PDB IDs in CSV: {len(pdb_ids)}")
    print(f"Copied files: {copied}")
    print(f"Missing files: {len(missing)}")

    if missing:
        print("\nMissing source files:")
        for path in missing:
            print(path)


if __name__ == "__main__":
    main()
