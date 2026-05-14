import csv
import os
from pathlib import Path

MODEL=os.environ.get("MODEL")  # Change to "best_ce" if you want to evaluate the best_ce model instead
    
SUMMARY_CSV = Path(f"/dapustor/nilufer/ADFLIP/INFERENCE/Results_first_neg/{MODEL}/summary.csv")
NEGATIVE_CSV = Path("/dapustor/nilufer/ADFLIP/INFERENCE/negative_sequences (1)noX.csv")

OUT_CSV = Path(f"/dapustor/nilufer/ADFLIP/INFERENCE/Results_first_neg/{MODEL}/SR_CRR.csv")
LOG_TXT = Path(f"/dapustor/nilufer/ADFLIP/INFERENCE/Results_first_neg/{MODEL}/SR_CRR.log")

NEG_COL = "Neg_seq_1"


def load_negative_data(path: Path):
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("Sequence_ID", "").strip()
            if seq_id:
                out[seq_id] = {
                    "gt_sequence": row.get("GT_sequence", "").strip(),
                    "neg_sequence": row.get(NEG_COL, "").strip(),
                }
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    negative_data = load_negative_data(NEGATIVE_CSV)

    rows = []
    log_lines = []

    seq_recoveries = []
    critical_recoveries = []
    perplexities = []

    missing_negative = []
    length_mismatches = []

    with open(SUMMARY_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_id = row.get("sequence_id", "").strip()
            s_true = row.get("s_true_chain", "").strip()
            s_pred = row.get("s_pred_chain", "").strip()
            perplexity = row.get("perplexity_overall", "").strip()

            result = {
                "sequence_id": seq_id,
                "gt_len": "",
                "neg_len": "",
                "s_true_len": len(s_true),
                "s_pred_len": len(s_pred),
                "perplexity_overall": perplexity,
                "overall_recovered": "",
                "sequence_recovery": "",
                "critical_count": "",
                "critical_recovered": "",
                "critical_recovery": "",
                "status": "",
                "note": "",
            }

            if seq_id not in negative_data:
                result["status"] = "missing_negative_csv_row"
                result["note"] = f"{seq_id} not found in {NEGATIVE_CSV}"
                rows.append(result)
                missing_negative.append(seq_id)
                continue

            gt_seq = negative_data[seq_id]["gt_sequence"]
            neg_seq = negative_data[seq_id]["neg_sequence"]

            result["gt_len"] = len(gt_seq)
            result["neg_len"] = len(neg_seq)

            if not gt_seq or not neg_seq:
                result["status"] = "missing_gt_or_neg"
                result["note"] = "GT_sequence or Neg_seq_1 is empty"
                rows.append(result)
                continue

            if len(gt_seq) != len(neg_seq) or len(gt_seq) != len(s_true) or len(s_true) != len(s_pred):
                result["status"] = "length_mismatch"
                result["note"] = (
                    f"GT len={len(gt_seq)}, Neg len={len(neg_seq)}, "
                    f"S_true len={len(s_true)}, S_pred len={len(s_pred)}"
                )
                rows.append(result)
                length_mismatches.append(seq_id)
                continue

            overall_recovered = sum(1 for a, b in zip(s_true, s_pred) if a == b)
            seq_rr = overall_recovered / len(s_true) if s_true else 0.0

            critical_positions = [i for i, (g, n) in enumerate(zip(gt_seq, neg_seq)) if g != n]
            critical_count = len(critical_positions)

            result["overall_recovered"] = overall_recovered
            result["sequence_recovery"] = f"{seq_rr:.6f}"

            if perplexity:
                try:
                    perplexities.append(float(perplexity))
                except ValueError:
                    pass

            if critical_count == 0:
                result["status"] = "no_critical_positions"
                result["critical_count"] = 0
                result["critical_recovered"] = 0
                result["critical_recovery"] = ""
                result["note"] = "GT_sequence and Neg_seq_1 are identical"
                rows.append(result)
                seq_recoveries.append(seq_rr)
                continue

            critical_recovered = sum(1 for i in critical_positions if s_pred[i] == s_true[i])
            critical_rr = critical_recovered / critical_count

            result["status"] = "ok"
            result["critical_count"] = critical_count
            result["critical_recovered"] = critical_recovered
            result["critical_recovery"] = f"{critical_rr:.6f}"
            rows.append(result)

            seq_recoveries.append(seq_rr)
            critical_recoveries.append(critical_rr)

            log_lines.append(f"Sequence_ID: {seq_id}")
            log_lines.append(f"  GT len: {len(gt_seq)}")
            log_lines.append(f"  Neg len: {len(neg_seq)}")
            log_lines.append(f"  S_true len: {len(s_true)}")
            log_lines.append(f"  S_pred len: {len(s_pred)}")
            log_lines.append(f"  Perplexity overall: {perplexity}")
            log_lines.append(f"  Overall recovered: {overall_recovered}")
            log_lines.append(f"  Sequence recovery: {seq_rr:.6f}")
            log_lines.append(f"  Critical count: {critical_count}")
            log_lines.append(f"  Critical recovered: {critical_recovered}")
            log_lines.append(f"  Critical recovery: {critical_rr:.6f}")
            log_lines.append("-" * 100)

    mean_seq_rr = mean(seq_recoveries)
    mean_critical_rr = mean(critical_recoveries)
    mean_perplexity = mean(perplexities)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "gt_len",
                "neg_len",
                "s_true_len",
                "s_pred_len",
                "perplexity_overall",
                "overall_recovered",
                "sequence_recovery",
                "critical_count",
                "critical_recovered",
                "critical_recovery",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        f"Negative column used: {NEG_COL}",
        f"Valid rows used for mean sequence recovery: {len(seq_recoveries)}",
        f"Mean sequence recovery: {mean_seq_rr:.6f}",
        f"Valid rows used for mean critical recovery: {len(critical_recoveries)}",
        f"Mean critical recovery: {mean_critical_rr:.6f}",
        f"Valid rows used for mean perplexity: {len(perplexities)}",
        f"Mean perplexity overall: {mean_perplexity:.6f}",
        f"Missing negative-data rows: {len(missing_negative)}",
        "Missing Sequence_IDs:",
    ]
    summary_lines += missing_negative if missing_negative else ["None"]
    summary_lines += [
        "-" * 100,
        f"Length mismatch count: {len(length_mismatches)}",
        "Length mismatch Sequence_IDs:",
    ]
    summary_lines += length_mismatches if length_mismatches else ["None"]
    summary_lines += [
        "-" * 100,
    ]

    LOG_TXT.write_text("\n".join(summary_lines + log_lines), encoding="utf-8")

    print(f"Wrote CSV to: {OUT_CSV}")
    print(f"Wrote log to: {LOG_TXT}")
    print(f"Mean sequence recovery: {mean_seq_rr:.6f}")
    print(f"Mean critical recovery: {mean_critical_rr:.6f}")
    print(f"Mean perplexity overall: {mean_perplexity:.6f}")


if __name__ == "__main__":
    main()
