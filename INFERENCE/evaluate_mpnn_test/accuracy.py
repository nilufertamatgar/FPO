import csv
import os
import random
from pathlib import Path

MODEL = os.environ["MODEL"]

INPUT_CSV = Path("/dapustor/nilufer/ADFLIP/INFERENCE/test_protein_mpnn_final_noX.csv")
LOGPROB_DIR = Path(f"/dapustor/nilufer/ADFLIP/INFERENCE/Results_test/{MODEL}/chain_log_prob_csvs")

OUT_CSV = Path(f"/dapustor/nilufer/ADFLIP/INFERENCE/Results_test/{MODEL}/gt_vs_random_neg_scores.csv")
LOG_TXT = Path(f"/dapustor/nilufer/ADFLIP/INFERENCE/Results_test/{MODEL}/accuracy.log")

RANDOM_SEED = 42

AA_TO_COL = {
    "A": "AA_A",
    "R": "AA_R",
    "N": "AA_N",
    "D": "AA_D",
    "C": "AA_C",
    "Q": "AA_Q",
    "E": "AA_E",
    "G": "AA_G",
    "H": "AA_H",
    "I": "AA_I",
    "L": "AA_L",
    "K": "AA_K",
    "M": "AA_M",
    "F": "AA_F",
    "P": "AA_P",
    "S": "AA_S",
    "T": "AA_T",
    "W": "AA_W",
    "Y": "AA_Y",
    "V": "AA_V",
    "X": "AA_X",
}


def score_sequence(logprob_rows, seq):
    total = 0.0
    for i, aa in enumerate(seq):
        col = AA_TO_COL.get(aa)
        if col is None:
            raise ValueError(f"Unsupported residue '{aa}' at position {i}")
        total += float(logprob_rows[i][col])
    return total


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def pick_random_negative(row, rng):
    neg_candidates = []
    for key, value in row.items():
        if key.startswith("Neg_seq_") and value and value.strip():
            neg_candidates.append((key, value.strip()))

    if not neg_candidates:
        return None, None

    chosen_col, chosen_seq = rng.choice(neg_candidates)
    return chosen_col, chosen_seq


def main():
    rng = random.Random(RANDOM_SEED)

    rows_out = []
    log_lines = []

    gt_scores = []
    neg_scores = []
    pairwise_correct = 0
    pairwise_total = 0
    ties = 0

    missing_log_csv = []
    length_mismatch = []
    missing_negative = []

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_id = row.get("Sequence_ID", "").strip()
            gt_seq = row.get("GT_sequence", "").strip()

            if not seq_id:
                continue

            neg_col, neg_seq = pick_random_negative(row, rng)

            logprob_csv = LOGPROB_DIR / f"{seq_id}.csv"

            result = {
                "sequence_id": seq_id,
                "chosen_negative_column": neg_col or "",
                "gt_len": len(gt_seq),
                "neg_len": len(neg_seq) if neg_seq else "",
                "logprob_len": "",
                "gt_score_sum": "",
                "neg_score_sum": "",
                "gt_score_mean": "",
                "neg_score_mean": "",
                "gt_gt_neg": "",
                "status": "",
                "note": "",
            }

            if not logprob_csv.exists():
                result["status"] = "missing_logprob_csv"
                result["note"] = str(logprob_csv)
                rows_out.append(result)
                missing_log_csv.append(seq_id)
                continue

            with open(logprob_csv, "r", newline="", encoding="utf-8") as lf:
                log_reader = csv.DictReader(lf)
                logprob_rows = list(log_reader)

            result["logprob_len"] = len(logprob_rows)

            if not gt_seq:
                result["status"] = "missing_gt_sequence"
                result["note"] = "GT_sequence is empty"
                rows_out.append(result)
                continue

            if not neg_seq:
                result["status"] = "missing_negative_sequence"
                result["note"] = "No available negative sequence columns with content"
                rows_out.append(result)
                missing_negative.append(seq_id)
                continue

            if len(gt_seq) != len(neg_seq) or len(gt_seq) != len(logprob_rows):
                result["status"] = "length_mismatch"
                result["note"] = (
                    f"GT len={len(gt_seq)}, Neg len={len(neg_seq)}, "
                    f"logprob rows={len(logprob_rows)}"
                )
                rows_out.append(result)
                length_mismatch.append(seq_id)
                continue

            try:
                gt_score = score_sequence(logprob_rows, gt_seq)
                neg_score = score_sequence(logprob_rows, neg_seq)
            except Exception as e:
                result["status"] = "scoring_error"
                result["note"] = repr(e)
                rows_out.append(result)
                continue

            gt_mean = gt_score / len(gt_seq)
            neg_mean = neg_score / len(neg_seq)

            result["gt_score_sum"] = f"{gt_score:.6f}"
            result["neg_score_sum"] = f"{neg_score:.6f}"
            result["gt_score_mean"] = f"{gt_mean:.6f}"
            result["neg_score_mean"] = f"{neg_mean:.6f}"
            result["gt_gt_neg"] = gt_score > neg_score
            result["status"] = "ok"

            rows_out.append(result)

            gt_scores.append(gt_score)
            neg_scores.append(neg_score)
            pairwise_total += 1

            if gt_score > neg_score:
                pairwise_correct += 1
            elif gt_score == neg_score:
                ties += 1

            log_lines.append(f"Sequence_ID: {seq_id}")
            log_lines.append(f"  Chosen negative column: {neg_col}")
            log_lines.append(f"  GT len: {len(gt_seq)}")
            log_lines.append(f"  Neg len: {len(neg_seq)}")
            log_lines.append(f"  Log-prob rows: {len(logprob_rows)}")
            log_lines.append(f"  GT score sum: {gt_score:.6f}")
            log_lines.append(f"  Neg score sum: {neg_score:.6f}")
            log_lines.append(f"  GT score mean: {gt_mean:.6f}")
            log_lines.append(f"  Neg score mean: {neg_mean:.6f}")
            log_lines.append(f"  GT > Neg: {gt_score > neg_score}")
            log_lines.append("-" * 100)

    avg_gt_score = mean(gt_scores)
    avg_neg_score = mean(neg_scores)
    pairwise_acc = pairwise_correct / pairwise_total if pairwise_total else 0.0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "chosen_negative_column",
                "gt_len",
                "neg_len",
                "logprob_len",
                "gt_score_sum",
                "neg_score_sum",
                "gt_score_mean",
                "neg_score_mean",
                "gt_gt_neg",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    summary_lines = [
        f"Random seed: {RANDOM_SEED}",
        f"Valid pairwise comparisons: {pairwise_total}",
        f"Pairwise correct (GT > Neg): {pairwise_correct}",
        f"Ties: {ties}",
        f"Pairwise accuracy (GT > Neg): {pairwise_acc:.6f}",
        f"Average GT score sum: {avg_gt_score:.6f}",
        f"Average Neg score sum: {avg_neg_score:.6f}",
        f"Missing log-prob CSV count: {len(missing_log_csv)}",
        "Missing log-prob CSV Sequence_IDs:",
    ]
    summary_lines += missing_log_csv if missing_log_csv else ["None"]
    summary_lines += [
        "-" * 100,
        f"Missing negative sequence count: {len(missing_negative)}",
        "Sequence_IDs with no available negatives:",
    ]
    summary_lines += missing_negative if missing_negative else ["None"]
    summary_lines += [
        "-" * 100,
        f"Length mismatch count: {len(length_mismatch)}",
        "Length mismatch Sequence_IDs:",
    ]
    summary_lines += length_mismatch if length_mismatch else ["None"]
    summary_lines += [
        "-" * 100,
    ]

    LOG_TXT.write_text("\n".join(summary_lines + log_lines), encoding="utf-8")

    print(f"Wrote CSV to: {OUT_CSV}")
    print(f"Wrote log to: {LOG_TXT}")
    print(f"Average GT score sum: {avg_gt_score:.6f}")
    print(f"Average Neg score sum: {avg_neg_score:.6f}")
    print(f"Pairwise accuracy (GT > Neg): {pairwise_acc:.6f}")


if __name__ == "__main__":
    main()
