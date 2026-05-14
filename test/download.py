import csv
import urllib.request
from pathlib import Path


CSV_PATH = Path("/dapustor/nilufer/ADFLIP/test/negative_sequences.csv")
OUT_DIR = Path("/dapustor/nilufer/ADFLIP/test/ligand_MPNN_pdb")


def download_pdb(pdb_id: str, out_dir: Path) -> bool:
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    out_path = out_dir / f"{pdb_id}.pdb"

    if out_path.exists():
        print(f"Already exists: {out_path}")
        return True

    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"Downloaded: {pdb_id}")
        return True
    except Exception as e:
        print(f"Failed: {pdb_id} -> {e}")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdb_ids = set()

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("Sequence_ID", "").strip()
            if not seq_id or "_" not in seq_id:
                continue
            pdb_id = seq_id.split("_", 1)[0].lower()
            pdb_ids.add(pdb_id)

    print(f"Unique PDB IDs found: {len(pdb_ids)}")

    success = 0
    failed = 0

    for pdb_id in sorted(pdb_ids):
        ok = download_pdb(pdb_id, OUT_DIR)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone.")
    print(f"Successful downloads: {success}")
    print(f"Failed downloads: {failed}")


if __name__ == "__main__":
    main()
