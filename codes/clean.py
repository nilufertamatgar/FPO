import csv
from pathlib import Path

INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/Finetune_data_1/valid_protein_mpnn_final_noX.csv")
REPORT_CSV = Path("/dapustor/nilufer/ADFLIP/finetuning_data/valid_chain_npz_alignment_report.csv")
OUTPUT_CSV = Path("/dapustor/nilufer/ADFLIP/Finetune_data_1/ADFLIP_valid.csv")


def load_ok_ids(report_csv: Path) -> set[str]:
    ok_ids = set()
    with open(report_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("sequence_id", "").strip()
            status = row.get("status", "").strip().lower()
            if seq_id and status == "ok":
                ok_ids.add(seq_id)
    return ok_ids


def filter_input_csv(input_csv: Path, ok_ids: set[str], output_csv: Path) -> None:
    kept = 0
    removed = 0

    with open(input_csv, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                seq_id = row.get("Sequence_ID", "").strip()
                if seq_id in ok_ids:
                    writer.writerow(row)
                    kept += 1
                else:
                    removed += 1

    print(f"Input CSV: {input_csv}")
    print(f"Report CSV: {REPORT_CSV}")
    print(f"Output CSV: {output_csv}")
    print(f"Kept rows: {kept}")
    print(f"Removed rows: {removed}")


if __name__ == "__main__":
    ok_ids = load_ok_ids(REPORT_CSV)
    filter_input_csv(INPUT_CSV, ok_ids, OUTPUT_CSV)
