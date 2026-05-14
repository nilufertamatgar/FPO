import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PMT_CSV = Path("/dapustor/nilufer/ADFLIP/pMT/pMT_stage_sequence_activity.csv")
LOGPROB_CSV = Path("/dapustor/nilufer/ADFLIP/pMT/extracted_log_probs_pMT/log_prob_csvs/pMT_ligand.csv")
MAPPING_CSV = Path("/dapustor/nilufer/ADFLIP/pMT/extracted_log_probs_pMT/logit_index_to_token.csv")

OUTPUT_PNG = Path("/dapustor/nilufer/ADFLIP/pMT/pMT_stagewise_ligand_scores.png")
OUTPUT_SCORE_CSV = Path("/dapustor/nilufer/ADFLIP/pMT/pMT_stage_sequence_activity_scored.csv")

WT_COLOR = "darkblue"
POS_COLOR = "darkorange"
NEG_COLOR = "lightskyblue"

JITTER_WIDTH = 0.16
POINT_SIZE = 24
POINT_ALPHA = 0.75
WT_SIZE = 240

random.seed(42)

AA_TO_COLUMN = {
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


def load_stage_activity_sequences(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append({
                "RowID": idx + 1,
                "Stage": row["Stage"].strip(),
                "Sequence": row["Sequence"].strip(),
                "Activity": float(row["Activity"]),
            })
    return rows


def load_logprob_table(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def validate_mapping_file(mapping_csv):
    seen = {}
    with open(mapping_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen[row["token"].strip()] = row["column_name"].strip()
    return seen


def sequence_score(sequence, logprob_rows):
    if len(sequence) != len(logprob_rows):
        raise ValueError(
            f"Sequence length {len(sequence)} does not match log-prob rows {len(logprob_rows)}"
        )

    values = []
    for i, aa in enumerate(sequence):
        if aa not in AA_TO_COLUMN:
            raise ValueError(f"Unsupported residue '{aa}' at position {i}")

        col = AA_TO_COLUMN[aa]
        values.append(float(logprob_rows[i][col]))

    # Average log-probability across sequence residues
    return sum(values) / len(values)


def stage_sort_key(stage_name):
    return int(stage_name.split()[1])


def save_score_csv(rows, output_csv):
    fieldnames = ["RowID", "Stage", "Sequence", "Activity", "Class", "Score"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_stage_activity_sequences(PMT_CSV)
    logprob_rows = load_logprob_table(LOGPROB_CSV)
    mapping_info = validate_mapping_file(MAPPING_CSV)

    for aa, col in AA_TO_COLUMN.items():
        if col not in logprob_rows[0]:
            raise ValueError(f"Missing column {col} in {LOGPROB_CSV}")

    if mapping_info.get("ALA") != "AA_A":
        print("Warning: mapping file is different than expected, continuing anyway.")

    for row in rows:
        row["Score"] = sequence_score(row["Sequence"], logprob_rows)
        row["Class"] = "Positive" if row["Activity"] > 0.25 else "Negative"

    save_score_csv(rows, OUTPUT_SCORE_CSV)

    wt_row = None
    for row in rows:
        if row["Stage"] == "Stage 1":
            wt_row = row
            break

    if wt_row is None:
        raise ValueError("Could not find WT row from Stage 1")

    wt_score = wt_row["Score"]

    stages = sorted({row["Stage"] for row in rows}, key=stage_sort_key)
    stage_to_x = {stage: i for i, stage in enumerate(stages)}

    plt.figure(figsize=(12, 6))

    # Plot every sequence with small x-jitter so overlapping points become visible
    for row in rows:
        base_x = stage_to_x[row["Stage"]]
        x = base_x + random.uniform(-JITTER_WIDTH, JITTER_WIDTH)
        color = POS_COLOR if row["Class"] == "Positive" else NEG_COLOR

        plt.scatter(
            x,
            row["Score"],
            marker="o",
            color=color,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            zorder=3,
        )

    # Plot WT score as a star in every stage
    for stage in stages:
        x = stage_to_x[stage]
        plt.scatter(
            x,
            wt_score,
            marker="*",
            color=WT_COLOR,
            s=WT_SIZE,
            edgecolor="black",
            zorder=5,
        )

    plt.xticks(np.arange(len(stages)), stages, rotation=45)
    plt.xlabel("Stage")
    plt.ylabel("Mean log-probability score")
    plt.title("Stage-wise pMT ligand scores")

    legend_elements = [
        Line2D(
            [0], [0],
            marker="*",
            linestyle="None",
            markerfacecolor=WT_COLOR,
            markeredgecolor="black",
            markersize=15,
            label="WT / Native",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=POS_COLOR,
            markersize=8,
            label="Positive (Activity > 0.25)",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=NEG_COLOR,
            markersize=8,
            label="Negative (Activity <= 0.25)",
        ),
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {OUTPUT_PNG}")
    print(f"Saved scored CSV to: {OUTPUT_SCORE_CSV}")
    print(f"WT score plotted in all stages: {wt_score:.6f}")
    print(f"Total sequences plotted: {len(rows)}")


if __name__ == "__main__":
    main()
