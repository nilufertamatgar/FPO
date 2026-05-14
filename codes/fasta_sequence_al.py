import csv
from pathlib import Path


CSV_PATH = Path("/dapustor/nilufer/ADFLIP/finetuning_data/train_protein_mpnn_final_noX.csv")
FASTA_PATH = Path("/dapustor/nilufer/ADFLIP/data/cluster/mmseqs/train_chains.fasta")

LOG_PATH = Path("/dapustor/nilufer/ADFLIP/finetuning_data/train_fasta_alignment_log.txt")
REPORT_CSV_PATH = Path("/dapustor/nilufer/ADFLIP/finetuning_data/train_fasta_alignment_report.csv")
MISMATCH_CSV_PATH = Path("/dapustor/nilufer/ADFLIP/finetuning_data/train_fasta_alignment_mismatches.csv")   


def read_fasta(fasta_path):
    seqs = {}
    current_name = None
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    seqs[current_name] = "".join(current_seq)
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_name is not None:
        seqs[current_name] = "".join(current_seq)

    return seqs


def first_diff_index(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def main():
    fasta_map = read_fasta(FASTA_PATH)

    matched = 0
    mismatched = 0
    missing_in_fasta = 0
    malformed = 0
    total_rows = 0

    detail_lines = []
    report_rows = []
    mismatch_rows = []

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            total_rows += 1
            seq_id = row.get("Sequence_ID", "").strip()
            gt_seq = row.get("GT_sequence", "").strip()

            report = {
                "row_num": row_num,
                "Sequence_ID": seq_id,
                "status": "",
                "csv_len": len(gt_seq) if gt_seq else "",
                "fasta_len": "",
                "first_diff_index": "",
                "csv_sequence": gt_seq,
                "fasta_sequence": "",
            }

            if not seq_id or not gt_seq:
                malformed += 1
                report["status"] = "malformed"
                report_rows.append(report)
                mismatch_rows.append(report.copy())

                detail_lines.append(f"[malformed] row={row_num} Sequence_ID={seq_id!r}")
                detail_lines.append("-" * 80)
                continue

            fasta_seq = fasta_map.get(seq_id)

            if fasta_seq is None:
                missing_in_fasta += 1
                report["status"] = "missing_in_fasta"
                report_rows.append(report)
                mismatch_rows.append(report.copy())

                detail_lines.append(f"[missing fasta] {seq_id}")
                detail_lines.append("-" * 80)
                continue

            report["fasta_len"] = len(fasta_seq)
            report["fasta_sequence"] = fasta_seq

            if fasta_seq == gt_seq:
                matched += 1
                report["status"] = "match"
                report["first_diff_index"] = -1
                report_rows.append(report)

                detail_lines.append(f"[match] {seq_id}")
            else:
                mismatched += 1
                diff_idx = first_diff_index(gt_seq, fasta_seq)
                report["status"] = "mismatch"
                report["first_diff_index"] = diff_idx
                report_rows.append(report)
                mismatch_rows.append(report.copy())

                detail_lines.append(f"[mismatch] {seq_id}")
                detail_lines.append(f"  csv len:   {len(gt_seq)}")
                detail_lines.append(f"  fasta len: {len(fasta_seq)}")
                detail_lines.append(f"  first diff index: {diff_idx}")
                detail_lines.append(f"  csv:   {gt_seq}")
                detail_lines.append(f"  fasta: {fasta_seq}")

            detail_lines.append("-" * 80)

    summary_lines = [
        f"CSV_PATH: {CSV_PATH}",
        f"FASTA_PATH: {FASTA_PATH}",
        f"LOG_PATH: {LOG_PATH}",
        f"REPORT_CSV_PATH: {REPORT_CSV_PATH}",
        f"MISMATCH_CSV_PATH: {MISMATCH_CSV_PATH}",
        "",
        "Summary",
        f"  total_rows: {total_rows}",
        f"  matched: {matched}",
        f"  mismatched: {mismatched}",
        f"  missing_in_fasta: {missing_in_fasta}",
        f"  malformed: {malformed}",
        f"  mismatch_like_total: {len(mismatch_rows)}",
        "",
        "=" * 80,
        "",
    ]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(summary_lines + detail_lines), encoding="utf-8")

    fieldnames = [
        "row_num",
        "Sequence_ID",
        "status",
        "csv_len",
        "fasta_len",
        "first_diff_index",
        "csv_sequence",
        "fasta_sequence",
    ]

    REPORT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    with open(MISMATCH_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mismatch_rows)

    print(f"Wrote log to: {LOG_PATH}")
    print(f"Wrote full report CSV to: {REPORT_CSV_PATH}")
    print(f"Wrote mismatch-only CSV to: {MISMATCH_CSV_PATH}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
