#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_BASE_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

_thread_local = threading.local()


def _get_session():
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "ADFLIP-downloader/1.0"})
        _thread_local.session = session
    return _thread_local.session


def _normalize_ids(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("pdb_ids", "pdb_id", "ids", "id"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unsupported JSON format: expected list of PDB IDs")


def _download_one(pdb_id, out_dir, base_url, overwrite, retries, timeout, subdir_prefix):
    pdb_id = str(pdb_id).strip()
    if not pdb_id:
        return (pdb_id, "skip-empty")

    pdb_id_lower = pdb_id.lower()
    pdb_id_upper = pdb_id.upper()

    if subdir_prefix:
        out_path = Path(out_dir) / pdb_id_lower[1:3] / f"{pdb_id_lower}.cif.gz"
    else:
        out_path = Path(out_dir) / f"{pdb_id_lower}.cif.gz"

    if out_path.exists() and not overwrite:
        return (pdb_id_lower, "skip-exists")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = base_url.format(pdb_id=pdb_id_upper)

    session = _get_session()
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, timeout=timeout, stream=True) as r:
                if r.status_code == 404:
                    return (pdb_id_lower, "404")
                r.raise_for_status()

                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, out_path)

            return (pdb_id_lower, "ok")

        except requests.HTTPError as e:
            last_err = f"HTTPError {e.response.status_code if e.response is not None else 'unknown'}"
        except requests.RequestException as e:
            last_err = f"RequestException {e}"
        except Exception as e:
            last_err = f"Error {e}"

        if attempt < retries:
            time.sleep(0.2)

    return (pdb_id_lower, f"fail: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Download PDB mmCIF files from a JSON list.")
    parser.add_argument("--json", required=True, help="Path to JSON file containing PDB IDs (list).")
    parser.add_argument("--out_dir", required=True, help="Output directory for .cif.gz files.")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="Download URL template.")
    parser.add_argument("--workers", type=int, default=32, help="Number of download workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per file.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds per request.")
    parser.add_argument("--subdir_prefix", action="store_true", help="Store files in subdir by ID[1:3].")

    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdb_ids = _normalize_ids(data)
    if not pdb_ids:
        print("No PDB IDs found.")
        return 1

    total = len(pdb_ids)
    print(f"Downloading {total} PDBs -> {args.out_dir}")

    ok = skip = fail = not_found = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                _download_one,
                pdb_id,
                args.out_dir,
                args.base_url,
                args.overwrite,
                args.retries,
                args.timeout,
                args.subdir_prefix,
            )
            for pdb_id in pdb_ids
        ]

        for fut in tqdm(concurrent.futures.as_completed(futures), total=total, desc="download"):
            pdb_id, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip-exists":
                skip += 1
            elif status == "404":
                not_found += 1
            else:
                fail += 1
                print(f"{pdb_id}: {status}")

    print(f"done. ok={ok} skip={skip} 404={not_found} fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
