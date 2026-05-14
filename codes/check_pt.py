import torch
from pathlib import Path

pt_path = Path("/dapustor/nilufer/ADFLIP/results/benchmark/ADFLIP_v1/pMT_adaptive_step_8_temp_0.1_noise_0.1_ther_0.9_argmax_1_ns_1_tfmr.pt")
out_path = Path("/dapustor/nilufer/ADFLIP/results/benchmark/ADFLIP_v1/pMT_adaptive.txt")

record = torch.load(pt_path, map_location="cpu")

processed_count = 0
not_processed_count = 0
processed_pdbs = []
not_processed_pdbs = []

lines = []
lines.append(f"Loaded file: {pt_path}")
lines.append(f"Top-level type: {type(record)}")
lines.append(f"Number of PDB entries: {len(record)}")
lines.append("")

for pdb_name, result in record.items():
    lines.append(f"PDB: {pdb_name}")
    lines.append(f"Entry type: {type(result)}")

    is_processed = False

    if isinstance(result, dict):
        lines.append(f"Keys: {list(result.keys())}")

        # Adjust this rule if your pipeline uses different required keys
        required_keys = {"S_true", "ensemble_logits"}
        if required_keys.issubset(result.keys()):
            is_processed = True

        for key, value in result.items():
            if torch.is_tensor(value):
                lines.append(
                    f"  {key}: tensor, shape={tuple(value.shape)}, dtype={value.dtype}"
                )
            elif isinstance(value, list):
                lines.append(f"  {key}: list, len={len(value)}")
                if len(value) > 0 and isinstance(value[0], (float, int, str)):
                    preview = value[:3]
                    lines.append(f"    preview: {preview}")
            else:
                lines.append(f"  {key}: {type(value)} -> {value}")
    else:
        lines.append(f"Value: {result}")

    if is_processed:
        processed_count += 1
        processed_pdbs.append(pdb_name)
        lines.append("Status: processed")
    else:
        not_processed_count += 1
        not_processed_pdbs.append(pdb_name)
        lines.append("Status: not processed")

    lines.append("")
    lines.append("-" * 80)
    lines.append("")

summary_lines = [
    f"Loaded file: {pt_path}",
    f"Top-level type: {type(record)}",
    f"Number of PDB entries: {len(record)}",
    f"Processed PDBs: {processed_count}",
    f"Not processed PDBs: {not_processed_count}",
    "",
    "Processed PDB list:",
    *processed_pdbs,
    "",
    "Not processed PDB list:",
    *not_processed_pdbs,
    "",
    "=" * 80,
    "",
]

out_path.write_text("\n".join(summary_lines + lines), encoding="utf-8")
print(f"Wrote summary to: {out_path}")
print(f"Processed PDBs: {processed_count}")
print(f"Not processed PDBs: {not_processed_count}")
