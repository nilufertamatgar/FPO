import csv
from pathlib import Path

from data.all_atom_parse import load_structure_data, index_to_token, restype_3to1


CSV_PATH = Path("/dapustor/nilufer/ADFLIP/finetuning_data/train_protein_mpnn_final.csv")
PARSED_ROOT = Path("/dapustor/nilufer/ADFLIP/dataset/train_parsed")
LOG_PATH = Path("/dapustor/nilufer/ADFLIP/finetuning_data/sequence_alignment_log.txt")


def extract_chain_sequence_from_npz(npz_path, chain_label):
    data = load_structure_data(str(npz_path))

    chain_map = getattr(data, "asym_id_to_chain_index", {})
    if chain_label not in chain_map:
        return None, f"chain label {chain_label} not found in asym_id_to_chain_index"

    chain_idx = chain_map[chain_label]

    center_mask = data.is_center & data.is_protein
    if hasattr(data, "backbone_mask"):
        center_mask = center_mask & data.backbone_mask

    chain_mask = data.chain_id == chain_idx
    mask = center_mask & chain_mask

    tokens = data.residue_token[mask]
    seq = "".join(restype_3to1.get(index_to_token[int(tok)], "X") for tok in tokens)
    return seq, None


def find_npz_for_pdb(parsed_root, pdb_id):
    subdir = pdb_id[1:3].lower()
    npz_path = parsed_root / subdir / f"{pdb_id.lower()}.npz"
    if npz_path.exists():
        return npz_path

    matches = list(parsed_root.rglob(f"{pdb_id.lower()}.npz"))
    if matches:
        return matches[0]

    return None


def log_line(lines, text):
    print(text)
    lines.append(text)


def main():
    checked = 0
    matched = 0
    mismatched = 0
    missing_npz = 0
    missing_chain = 0

    lines = []
    log_line(lines, f"CSV_PATH: {CSV_PATH}")
    log_line(lines, f"PARSED_ROOT: {PARSED_ROOT}")
    log_line(lines, "")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_id = row["Sequence_ID"].strip()
            gt_seq = row["GT_sequence"].strip()

            if "_" not in seq_id:
                log_line(lines, f"[skip] malformed Sequence_ID: {seq_id}")
                continue

            pdb_id, chain_label = seq_id.split("_", 1)
            npz_path = find_npz_for_pdb(PARSED_ROOT, pdb_id)

            if npz_path is None:
                missing_npz += 1
                log_line(lines, f"[missing npz] {seq_id}")
                continue

            parsed_seq, err = extract_chain_sequence_from_npz(npz_path, chain_label)
            if parsed_seq is None:
                missing_chain += 1
                log_line(lines, f"[missing chain] {seq_id} -> {err}")
                continue

            checked += 1
            if parsed_seq == gt_seq:
                matched += 1
                log_line(lines, f"[match] {seq_id}")
            else:
                mismatched += 1
                log_line(lines, f"[mismatch] {seq_id}")
                log_line(lines, f"  npz: {npz_path}")
                log_line(lines, f"  csv len:    {len(gt_seq)}")
                log_line(lines, f"  parsed len: {len(parsed_seq)}")
                log_line(lines, f"  csv:    {gt_seq}")
                log_line(lines, f"  parsed: {parsed_seq}")

            log_line(lines, "-" * 80)

            if checked >= 50:
                break

    log_line(lines, "")
    log_line(lines, "Summary")
    log_line(lines, f"  checked: {checked}")
    log_line(lines, f"  matched: {matched}")
    log_line(lines, f"  mismatched: {mismatched}")
    log_line(lines, f"  missing_npz: {missing_npz}")
    log_line(lines, f"  missing_chain: {missing_chain}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote log to: {LOG_PATH}")


if __name__ == "__main__":
    main()
