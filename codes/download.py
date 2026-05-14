import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


DEFAULT_BASE_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

SRC_DIRS = [
    Path("/dapustor/nilufer/ADFLIP/test/PDB_input/test_multi_chain"),
    Path("/dapustor/nilufer/ADFLIP/test/PDB_input/test_single_chain"),
]

OUT_DIR = Path("/dapustor/nilufer/ADFLIP/dataset/test_MPNN")
MAX_WORKERS = 16


def extract_pdb_id(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".cif.gz"):
        return name[:-7]
    if name.endswith(".cif"):
        return name[:-4]
    if name.endswith(".pdb"):
        return name[:-4]
    return path.stem.lower()


def collect_pdb_ids(src_dirs):
    pdb_ids = set()
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        for p in src_dir.iterdir():
            if p.is_file():
                pdb_ids.add(extract_pdb_id(p))
    return sorted(pdb_ids)


def download_one(pdb_id: str):
    url = DEFAULT_BASE_URL.format(pdb_id=pdb_id.upper())
    out_path = OUT_DIR / f"{pdb_id.lower()}.cif.gz"

    if out_path.exists():
        return pdb_id, "exists", str(out_path)

    try:
        urllib.request.urlretrieve(url, out_path)
        return pdb_id, "downloaded", str(out_path)
    except Exception as e:
        return pdb_id, "failed", repr(e)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdb_ids = collect_pdb_ids(SRC_DIRS)

    print(f"Unique PDB IDs found: {len(pdb_ids)}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, pdb_id) for pdb_id in pdb_ids]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading CIFs"):
            results.append(future.result())

    downloaded = [r for r in results if r[1] == "downloaded"]
    exists = [r for r in results if r[1] == "exists"]
    failed = [r for r in results if r[1] == "failed"]

    print(f"\nDownloaded: {len(downloaded)}")
    print(f"Already existed: {len(exists)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed PDB IDs:")
        for pdb_id, _, err in failed:
            print(f"{pdb_id}: {err}")


if __name__ == "__main__":
    main()
