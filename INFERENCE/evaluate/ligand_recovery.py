import math
from pathlib import Path

import torch


PT_PATH = Path(
    "/dapustor/nilufer/ADFLIP/results/benchmark/beta0.2alpha1_lr0.001_ADFLIP_yyh_best/"
    "test_set_109_adaptive_step_8_temp_0.1_noise_0.1_ther_0.9_argmax_1_ns_1_tfmr.pt"
)

OUT_LOG = PT_PATH.with_name(PT_PATH.stem + "_summary.log")


def to_scalar(value):
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def safe_mean(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def fmt_float(x):
    if x is None:
        return "None"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.6f}"


def main():
    data = torch.load(PT_PATH, map_location="cpu")

    rr_vals = []
    non_protein_rr_vals = []
    ion_rr_vals = []
    nucleotide_rr_vals = []
    molecule_rr_vals = []

    per_pdb_lines = []

    per_pdb_lines.append(f"Loaded file: {PT_PATH}")
    per_pdb_lines.append(f"Top-level type: {type(data)}")
    per_pdb_lines.append(f"Number of PDB entries: {len(data)}")
    per_pdb_lines.append("")

    for pdb_id in sorted(data.keys()):
        entry = data[pdb_id]

        rr = to_scalar(entry.get("rr"))
        non_protein_rr = to_scalar(entry.get("non_protein_rr"))
        ion_rr = to_scalar(entry.get("ion_rr"))
        nucleotide_rr = to_scalar(entry.get("nucleotide_rr"))
        molecule_rr = to_scalar(entry.get("molecule_rr"))

        s_pred = entry.get("S_pred", "")
        if isinstance(s_pred, list):
            s_pred = s_pred[0] if s_pred else ""

        s_true = entry.get("S_true", "")

        rr_vals.append(rr)
        non_protein_rr_vals.append(non_protein_rr)
        ion_rr_vals.append(ion_rr)
        nucleotide_rr_vals.append(nucleotide_rr)
        molecule_rr_vals.append(molecule_rr)

        per_pdb_lines.append(f"PDB: {pdb_id}")
        per_pdb_lines.append(f"  rr: {fmt_float(rr)}")
        per_pdb_lines.append(f"  non_protein_rr: {fmt_float(non_protein_rr)}")
        per_pdb_lines.append(f"  ion_rr: {fmt_float(ion_rr)}")
        per_pdb_lines.append(f"  nucleotide_rr: {fmt_float(nucleotide_rr)}")
        per_pdb_lines.append(f"  molecule_rr: {fmt_float(molecule_rr)}")
        per_pdb_lines.append(f"  S_true: {s_true}")
        per_pdb_lines.append(f"  S_pred: {s_pred}")
        per_pdb_lines.append("-" * 100)

    summary_lines = [
        f"Loaded file: {PT_PATH}",
        f"Number of PDB entries: {len(data)}",
        f"Mean rr: {fmt_float(safe_mean(rr_vals))}",
        f"Mean non_protein_rr: {fmt_float(safe_mean(non_protein_rr_vals))}",
        f"Mean ion_rr: {fmt_float(safe_mean(ion_rr_vals))}",
        f"Mean nucleotide_rr: {fmt_float(safe_mean(nucleotide_rr_vals))}",
        f"Mean molecule_rr: {fmt_float(safe_mean(molecule_rr_vals))}",
        "=" * 100,
        "",
    ]

    OUT_LOG.write_text("\n".join(summary_lines + per_pdb_lines), encoding="utf-8")

    print(f"Wrote log to: {OUT_LOG}")
    print(f"Mean rr: {fmt_float(safe_mean(rr_vals))}")
    print(f"Mean non_protein_rr: {fmt_float(safe_mean(non_protein_rr_vals))}")
    print(f"Mean ion_rr: {fmt_float(safe_mean(ion_rr_vals))}")
    print(f"Mean nucleotide_rr: {fmt_float(safe_mean(nucleotide_rr_vals))}")
    print(f"Mean molecule_rr: {fmt_float(safe_mean(molecule_rr_vals))}")


if __name__ == "__main__":
    main()
