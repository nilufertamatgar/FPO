import csv
from pathlib import Path

INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/INFERENCE/negative_sequences (1).csv")
OUTPUT_CSV = Path("/dapustor/nilufer/ADFLIP/INFERENCE/negative_sequences (1)noX.csv")
BAD_ROWS_CSV = Path("/dapustor/nilufer/ADFLIP/INFERENCE/negative_sequences_noX_bad_rows.csv")   


def clean_sequence_by_gt_x(gt_seq: str, seq: str) -> str:
    keep_indices = [i for i, aa in enumerate(gt_seq) if aa != "X"]
    return "".join(seq[i] for i in keep_indices)


def main():
    processed = 0
    written = 0
    skipped = 0

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        bad_fieldnames = ["Sequence_ID", "reason"]

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out, open(
            BAD_ROWS_CSV, "w", newline="", encoding="utf-8"
        ) as f_bad:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            bad_writer = csv.DictWriter(f_bad, fieldnames=bad_fieldnames)

            writer.writeheader()
            bad_writer.writeheader()

            for row in reader:
                processed += 1
                seq_id = row.get("Sequence_ID", "").strip()
                gt_seq = row.get("GT_sequence", "").strip()

                if not gt_seq:
                    skipped += 1
                    bad_writer.writerow(
                        {"Sequence_ID": seq_id, "reason": "empty GT_sequence"}
                    )
                    continue

                neg_keys = [k for k in row if k.startswith("Neg_seq_")]

                all_sequences = {"GT_sequence": gt_seq}
                for key in neg_keys:
                    all_sequences[key] = row.get(key, "").strip()

                original_bad = [
                    key for key, seq in all_sequences.items()
                    if seq and len(seq) != len(gt_seq)
                ]
                if original_bad:
                    skipped += 1
                    bad_writer.writerow(
                        {
                            "Sequence_ID": seq_id,
                            "reason": (
                                "original length mismatch with GT_sequence in: "
                                + ", ".join(original_bad)
                            ),
                        }
                    )
                    continue

                keep_indices = [i for i, aa in enumerate(gt_seq) if aa != "X"]
                cleaned_gt = "".join(gt_seq[i] for i in keep_indices)

                new_row = dict(row)
                new_row["GT_sequence"] = cleaned_gt

                cleaned_lengths_bad = []

                for key in neg_keys:
                    neg_seq = row.get(key, "").strip()
                    if not neg_seq:
                        new_row[key] = neg_seq
                        continue

                    cleaned_neg = "".join(neg_seq[i] for i in keep_indices)
                    new_row[key] = cleaned_neg

                    if len(cleaned_neg) != len(cleaned_gt):
                        cleaned_lengths_bad.append(key)

                if cleaned_lengths_bad:
                    skipped += 1
                    bad_writer.writerow(
                        {
                            "Sequence_ID": seq_id,
                            "reason": (
                                "cleaned length mismatch with cleaned GT_sequence in: "
                                + ", ".join(cleaned_lengths_bad)
                            ),
                        }
                    )
                    continue

                writer.writerow(new_row)
                written += 1

    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Bad rows CSV: {BAD_ROWS_CSV}")
    print(f"Processed rows: {processed}")
    print(f"Written rows: {written}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
