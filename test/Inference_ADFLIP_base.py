import csv
import re
import subprocess
from pathlib import Path
from datetime import datetime


SCRIPT_PATH = Path("/dapustor/nilufer/ADFLIP/test/design.py")
PDB_DIR = Path("/dapustor/nilufer/ADFLIP/test/ligand_MPNN_pdb")
INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/test/negative_sequences.csv")

OUT_CSV = Path("/dapustor/nilufer/ADFLIP/test/base_ligand_mpnn_pdb.csv")

LOG_DIR = Path("/dapustor/nilufer/ADFLIP/test/design_batch_logs_base")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"design_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

CKPT_PATH = "/dapustor/nilufer/ADFLIP/results/weights/ADFLIP_v1.pt"
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


def load_sequence_ids(csv_path: Path):
    rows = []
    seen = set()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("Sequence_ID", "").strip()
            if not seq_id or "_" not in seq_id:
                continue
            if seq_id in seen:
                continue
            seen.add(seq_id)

            pdb_id = seq_id.split("_", 1)[0].lower()
            pdb_path = PDB_DIR / f"{pdb_id}.pdb"

            rows.append(
                {
                    "sequence_id": seq_id,
                    "pdb_id": pdb_id,
                    "pdb_path": pdb_path,
                }
            )

    return rows


def run_one(sequence_id: str, pdb_path: Path):
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

    if not pdb_path.exists():
        log("=" * 100)
        log(f"Sequence_ID: {sequence_id}")
        log(f"PDB missing: {pdb_path}")
        log("=" * 100)
        return {
            "sequence_id": sequence_id,
            "pdb_file": pdb_path.name,
            "returncode": "",
            "final_rr": "",
            "overall_rr": "",
            "interacting_rr": "",
            "generated_sequence": "",
            "note": "missing_pdb",
        }

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/dapustor/nilufer/ADFLIP/test",
    )

    metrics = parse_metrics(proc.stdout)

    log("=" * 100)
    log(f"Sequence_ID: {sequence_id}")
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
        "sequence_id": sequence_id,
        "pdb_file": pdb_path.name,
        "returncode": proc.returncode,
        "final_rr": metrics["final_rr"],
        "overall_rr": metrics["overall_rr"],
        "interacting_rr": metrics["interacting_rr"],
        "generated_sequence": metrics["generated_sequence"],
        "note": "",
    }


def main():
    log(f"Log file: {LOG_PATH}")
    entries = load_sequence_ids(INPUT_CSV)
    log(f"Found {len(entries)} unique Sequence_ID entries")

    rows = []
    for entry in entries:
        rows.append(run_one(entry["sequence_id"], entry["pdb_path"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "pdb_file",
                "returncode",
                "final_rr",
                "overall_rr",
                "interacting_rr",
                "generated_sequence",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    log(f"Wrote CSV: {OUT_CSV}")
    log("Batch design run complete")


if __name__ == "__main__":
    main()
