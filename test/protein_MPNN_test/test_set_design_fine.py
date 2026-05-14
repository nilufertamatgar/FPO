import csv
import re
import subprocess
from pathlib import Path
from datetime import datetime


SCRIPT_PATH = Path("/dapustor/nilufer/ADFLIP/test/design.py")
SINGLE_DIR = Path("/dapustor/nilufer/ADFLIP/test/PDB_input/test_single_chain")
MULTI_DIR = Path("/dapustor/nilufer/ADFLIP/test/PDB_input/test_multi_chain")

OUT_SINGLE = Path("/dapustor/nilufer/ADFLIP/test/best_1_single_chain.csv")
OUT_MULTI = Path("/dapustor/nilufer/ADFLIP/test/best_1_multi_chain.csv")

LOG_DIR = Path("/dapustor/nilufer/ADFLIP/test/design_batch_logs_best_1")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"design_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

CKPT_PATH = "/dapustor/nilufer/ADFLIP/results/weights/best_1.pt"
DEVICE = "cuda:0"
METHOD = "adaptive"
DT = "0.2"
STEPS = "8"
THRESHOLD = "0.9"


def log(msg):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def list_inputs(folder: Path):
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and (p.suffix in {".pdb", ".cif"} or p.name.endswith(".cif.gz")):
            files.append(p)
    return files


def extract_metric(pattern, text):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def parse_metrics(stdout: str):
    return {
        "final_rr": extract_metric(r"final rr:\s*([0-9.]+)", stdout),
        "overall_rr": extract_metric(r"Overall residue recovery rate:\s*([0-9.]+)", stdout),
        "interacting_rr": extract_metric(r"Interacting-residue recovery rate:\s*([0-9.]+)", stdout),
        "generated_sequence": extract_metric(r"Generated sequence:\s*(\S+)", stdout),
    }


def run_one(pdb_path: Path, dataset_name: str):
    cmd = [
        "python",
        str(SCRIPT_PATH),
        "--pdb", str(pdb_path),
        "--ckpt", CKPT_PATH,
        "--device", DEVICE,
        "--method", METHOD,
        "--dt", DT,
        "--steps", STEPS,
        "--threshold", THRESHOLD,
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/dapustor/nilufer/ADFLIP/test",
    )

    metrics = parse_metrics(proc.stdout)

    log("=" * 100)
    log(f"DATASET: {dataset_name}")
    log(f"PDB: {pdb_path.name}")
    log(f"COMMAND: {' '.join(cmd)}")
    log(f"RETURNCODE: {proc.returncode}")

    if proc.stdout:
        log("STDOUT:")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(proc.stdout.rstrip() + "\n")

    if proc.stderr:
        log("STDERR:")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(proc.stderr.rstrip() + "\n")

    log("=" * 100)

    return {
        "pdb_file": pdb_path.name,
        "returncode": proc.returncode,
        "final_rr": metrics["final_rr"],
        "overall_rr": metrics["overall_rr"],
        "interacting_rr": metrics["interacting_rr"],
        "generated_sequence": metrics["generated_sequence"],
    }


def run_folder(folder: Path, out_csv: Path, dataset_name: str):
    rows = []
    pdb_files = list_inputs(folder)

    log(f"Processing folder: {folder}")
    log(f"Found {len(pdb_files)} input files")

    for pdb_path in pdb_files:
        rows.append(run_one(pdb_path, dataset_name))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_file",
                "returncode",
                "final_rr",
                "overall_rr",
                "interacting_rr",
                "generated_sequence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote CSV: {out_csv}")


def main():
    log(f"Log file: {LOG_PATH}")
    run_folder(SINGLE_DIR, OUT_SINGLE, "single_chain")
    run_folder(MULTI_DIR, OUT_MULTI, "multi_chain")
    log("Batch design run complete")


if __name__ == "__main__":
    main()
