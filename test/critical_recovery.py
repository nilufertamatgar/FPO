import csv
from pathlib import Path

MODEL = "lambda_1_epoch_9"  # Used for naming output files and log

PREFERENCE_CSV = Path("/dapustor/nilufer/ADFLIP/test/negative_sequences_noX.csv")
# GENERATED_CSV = Path(f"/dapustor/nilufer/ADFLIP/test/{MODEL}_ligand_mpnn_pdb.csv")
GENERATED_CSV = Path(f"/dapustor/nilufer/ADFLIP/test_yyh/beta0.2alpha0.5_yyh900.csv")

# OUT_CSV = Path(f"/dapustor/nilufer/ADFLIP/test/{MODEL}_single_chain.csv")
# LOG_TXT = Path(f"/dapustor/nilufer/ADFLIP/test/{MODEL}_single_chain.log")
OUT_CSV = Path(f"/dapustor/nilufer/ADFLIP/test_yyh/beta0.2alpha0.5_yyh900.csv")
LOG_TXT = Path(f"/dapustor/nilufer/ADFLIP/test_yyh/beta0.2alpha0.5_yyh900.log")


NEG_COL = "Neg_seq_1"


def load_generated_sequences(path: Path):
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_file = row["pdb_file"].strip()
            pdb_id = Path(pdb_file).stem.lower()
            out[pdb_id] = {
                "generated_sequence": row.get("generated_sequence", "").strip(),
                "interacting_rr": row.get("interacting_rr", "").strip(),
                "overall_rr": row.get("overall_rr", "").strip(),
                "final_rr": row.get("final_rr", "").strip(),
            }
    return out


def main():
    generated = load_generated_sequences(GENERATED_CSV)
    rows = []
    log_lines = []

    valid_rrs = []
    overall_rrs = []
    interacting_rrs = []
    gt_generated_length_mismatch_ids = []
    gt_negative_length_mismatch_ids = []

    with open(PREFERENCE_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_id = row.get("Sequence_ID", "").strip()
            gt_seq = row.get("GT_sequence", "").strip()
            neg_seq = row.get(NEG_COL, "").strip()

            if not seq_id or "_" not in seq_id:
                continue

            pdb_id = seq_id.split("_", 1)[0].lower()
            gen_info = generated.get(pdb_id, {})
            gen_seq = gen_info.get("generated_sequence", "")
            interacting_rr = gen_info.get("interacting_rr", "")
            model_overall_rr = gen_info.get("overall_rr", "")
            final_rr = gen_info.get("final_rr", "")

            result = {
                "sequence_id": seq_id,
                "pdb_id": pdb_id,
                "gt_len": len(gt_seq),
                "neg_len": len(neg_seq),
                "generated_len": len(gen_seq) if gen_seq else "",
                "interacting_rr": interacting_rr,
                "model_overall_rr": model_overall_rr,
                "final_rr": final_rr,
                "critical_count": "",
                "critical_recovered": "",
                "critical_set_rr": "",
                "overall_recovered": "",
                "overall_seq_rr": "",
                "status": "",
                "note": "",
            }

            if not gen_seq:
                result["status"] = "missing_generated"
                result["note"] = f"No generated sequence found for pdb_id={pdb_id}"
                rows.append(result)
                continue

            if not neg_seq:
                result["status"] = "missing_negative"
                result["note"] = f"Empty negative sequence in {NEG_COL}"
                rows.append(result)
                continue

            if len(gt_seq) != len(gen_seq):
                result["status"] = "gt_generated_length_mismatch"
                result["note"] = f"GT len={len(gt_seq)}, generated len={len(gen_seq)}"
                rows.append(result)
                gt_generated_length_mismatch_ids.append(seq_id)
                continue

            if len(gt_seq) != len(neg_seq):
                result["status"] = "gt_negative_length_mismatch"
                result["note"] = f"GT len={len(gt_seq)}, negative len={len(neg_seq)}"
                rows.append(result)
                gt_negative_length_mismatch_ids.append(seq_id)
                continue

            overall_recovered = sum(1 for g, s in zip(gt_seq, gen_seq) if g == s)
            overall_seq_rr = overall_recovered / len(gt_seq) if gt_seq else 0.0

            critical_positions = [
                i for i, (g, n) in enumerate(zip(gt_seq, neg_seq)) if g != n
            ]
            critical_count = len(critical_positions)

            result["overall_recovered"] = overall_recovered
            result["overall_seq_rr"] = f"{overall_seq_rr:.6f}"

            if critical_count == 0:
                result["status"] = "no_critical_positions"
                result["critical_count"] = 0
                result["critical_recovered"] = 0
                result["critical_set_rr"] = ""
                result["note"] = "GT and negative sequence are identical"
                rows.append(result)
                overall_rrs.append(overall_seq_rr)
                if interacting_rr:
                    interacting_rrs.append(float(interacting_rr))
                continue

            critical_recovered = sum(
                1 for i in critical_positions if gen_seq[i] == gt_seq[i]
            )
            critical_rr = critical_recovered / critical_count

            result["status"] = "ok"
            result["critical_count"] = critical_count
            result["critical_recovered"] = critical_recovered
            result["critical_set_rr"] = f"{critical_rr:.6f}"
            rows.append(result)

            valid_rrs.append(critical_rr)
            overall_rrs.append(overall_seq_rr)
            if interacting_rr:
                interacting_rrs.append(float(interacting_rr))

            log_lines.append(f"Sequence_ID: {seq_id}")
            log_lines.append(f"  GT len: {len(gt_seq)}")
            log_lines.append(f"  Negative len: {len(neg_seq)}")
            log_lines.append(f"  Generated len: {len(gen_seq)}")
            log_lines.append(f"  final_rr: {final_rr}")
            log_lines.append(f"  model overall_rr: {model_overall_rr}")
            log_lines.append(f"  interacting_rr: {interacting_rr}")
            log_lines.append(f"  Overall recovered: {overall_recovered}")
            log_lines.append(f"  Overall sequence RR: {overall_seq_rr:.6f}")
            log_lines.append(f"  Critical count: {critical_count}")
            log_lines.append(f"  Critical recovered: {critical_recovered}")
            log_lines.append(f"  Critical set RR: {critical_rr:.6f}")
            log_lines.append("-" * 100)

    mean_critical_rr = sum(valid_rrs) / len(valid_rrs) if valid_rrs else 0.0
    mean_overall_rr = sum(overall_rrs) / len(overall_rrs) if overall_rrs else 0.0
    mean_interacting_rr = sum(interacting_rrs) / len(interacting_rrs) if interacting_rrs else 0.0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_id",
                "pdb_id",
                "gt_len",
                "neg_len",
                "generated_len",
                "interacting_rr",
                "model_overall_rr",
                "final_rr",
                "critical_count",
                "critical_recovered",
                "critical_set_rr",
                "overall_recovered",
                "overall_seq_rr",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        f"Negative column used: {NEG_COL}",
        f"Valid rows used for mean critical RR: {len(valid_rrs)}",
        f"Mean critical set recovery rate: {mean_critical_rr:.6f}",
        f"Mean overall sequence recovery rate: {mean_overall_rr:.6f}",
        f"Mean interacting residue recovery rate: {mean_interacting_rr:.6f}",
        f"GT-vs-generated length mismatch count: {len(gt_generated_length_mismatch_ids)}",
        "Sequence_IDs with GT-vs-generated length mismatch:",
    ]
    summary_lines += gt_generated_length_mismatch_ids if gt_generated_length_mismatch_ids else ["None"]
    summary_lines += [
        "-" * 100,
        f"GT-vs-negative length mismatch count: {len(gt_negative_length_mismatch_ids)}",
        "Sequence_IDs with GT-vs-negative length mismatch:",
    ]
    summary_lines += gt_negative_length_mismatch_ids if gt_negative_length_mismatch_ids else ["None"]
    summary_lines += [
        "-" * 100,
    ]

    LOG_TXT.write_text("\n".join(summary_lines + log_lines), encoding="utf-8")

    print(f"Wrote CSV to: {OUT_CSV}")
    print(f"Wrote log to: {LOG_TXT}")
    print(f"Mean critical set recovery rate using {NEG_COL}: {mean_critical_rr:.6f}")
    print(f"Mean overall sequence recovery rate: {mean_overall_rr:.6f}")
    print(f"Mean interacting residue recovery rate: {mean_interacting_rr:.6f}")
    print(f"GT-vs-generated length mismatch count: {len(gt_generated_length_mismatch_ids)}")


if __name__ == "__main__":
    main()
