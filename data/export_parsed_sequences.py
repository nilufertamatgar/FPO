#!/usr/bin/env python3
import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_token_tables() -> Tuple[Dict[int, str], Dict[str, str], str]:
    """
    Load residue-token tables from parser modules.
    We try the Biopython parser first, then the legacy parser.
    """
    module_names = [
        "data.all_atom_parse_biopython",
        "data.all_atom_parse",
    ]
    errors = []
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            return mod.index_to_token, mod.restype_3to1, mod_name
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{mod_name}: {exc}")
    raise RuntimeError(
        "Failed to import parser token tables. "
        "Tried modules:\n" + "\n".join(errors)
    )


def _list_npz_files(parsed_dir: Path) -> List[Path]:
    return sorted(parsed_dir.rglob("*.npz"))


def _as_python_obj(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def _coerce_chain_index_map(npz_obj) -> Dict[int, str]:
    if "asym_id_to_chain_index" not in npz_obj.files:
        return {}

    raw = _as_python_obj(npz_obj["asym_id_to_chain_index"])
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        raw = _as_python_obj(raw[0])

    if not isinstance(raw, dict):
        return {}

    out: Dict[int, str] = {}
    for asym_id, idx in raw.items():
        try:
            out[int(idx)] = str(asym_id)
        except Exception:
            continue
    return out


def _coerce_assemblies(npz_obj) -> Tuple[List[str], List[List[int]]]:
    if "asmb_ids" not in npz_obj.files or "asmb_chain_indices" not in npz_obj.files:
        return [], []

    raw_ids = npz_obj["asmb_ids"]
    raw_chain_indices = npz_obj["asmb_chain_indices"]

    asmb_ids = [str(x) for x in np.asarray(raw_ids).tolist()]
    asmb_chain_indices = []
    for item in np.asarray(raw_chain_indices, dtype=object).tolist():
        if item is None:
            asmb_chain_indices.append([])
            continue
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if not isinstance(item, (list, tuple)):
            asmb_chain_indices.append([])
            continue
        cleaned = []
        for v in item:
            try:
                cleaned.append(int(v))
            except Exception:
                continue
        asmb_chain_indices.append(cleaned)

    return asmb_ids, asmb_chain_indices


def _iter_chain_tokens(
    residue_index: np.ndarray,
    residue_token: np.ndarray,
    chain_id: np.ndarray,
    is_protein: np.ndarray,
) -> Dict[int, List[int]]:
    chain_tokens: Dict[int, List[int]] = {}
    seen_residues = set()

    for rid, tok, cid, is_prot in zip(residue_index, residue_token, chain_id, is_protein):
        if not bool(is_prot):
            continue
        key = (int(cid), int(rid))
        if key in seen_residues:
            continue
        seen_residues.add(key)
        chain_tokens.setdefault(int(cid), []).append(int(tok))

    return chain_tokens


def _tokens_to_sequence(tokens: Iterable[int], index_to_token, restype_3to1) -> str:
    seq_chars = []
    for tok in tokens:
        residue_name = index_to_token.get(int(tok), "<UNK>")
        seq_chars.append(restype_3to1.get(residue_name, "X"))
    return "".join(seq_chars)


def export_sequences(
    parsed_dir: Path,
    fasta_out: Path,
    metadata_out: Path,
    min_len: int,
) -> None:
    index_to_token, restype_3to1, token_module = _load_token_tables()
    npz_files = _list_npz_files(parsed_dir)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under: {parsed_dir}")

    fasta_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open(fasta_out, "w") as fasta_f, open(metadata_out, "w", newline="") as meta_f:
        writer = csv.DictWriter(
            meta_f,
            fieldnames=[
                "chain_key",
                "pdb_id",
                "chain_index",
                "chain_label",
                "length",
                "unknown_count",
                "unknown_frac",
                "assemblies",
                "npz_path",
                "token_module",
            ],
        )
        writer.writeheader()

        for npz_path in tqdm(npz_files, desc="Exporting sequences"):
            with np.load(npz_path, allow_pickle=True) as d:
                required = {"residue_index", "residue_token", "chain_id", "is_protein"}
                if not required.issubset(set(d.files)):
                    continue

                chain_idx_to_label = _coerce_chain_index_map(d)
                asmb_ids, asmb_chain_indices = _coerce_assemblies(d)

                chain_tokens = _iter_chain_tokens(
                    d["residue_index"],
                    d["residue_token"],
                    d["chain_id"],
                    d["is_protein"],
                )

                pdb_id = npz_path.stem.lower()
                for chain_idx, tokens in chain_tokens.items():
                    seq = _tokens_to_sequence(tokens, index_to_token, restype_3to1)
                    if len(seq) < min_len:
                        continue

                    chain_label = chain_idx_to_label.get(chain_idx, str(chain_idx))
                    chain_key = f"{pdb_id}_{chain_label}"

                    chain_asmb_ids = []
                    for i, idxs in enumerate(asmb_chain_indices):
                        if chain_idx in idxs and i < len(asmb_ids):
                            chain_asmb_ids.append(asmb_ids[i])

                    unknown_count = seq.count("X")
                    unknown_frac = unknown_count / max(1, len(seq))

                    fasta_f.write(f">{chain_key}\n{seq}\n")
                    writer.writerow(
                        {
                            "chain_key": chain_key,
                            "pdb_id": pdb_id,
                            "chain_index": chain_idx,
                            "chain_label": chain_label,
                            "length": len(seq),
                            "unknown_count": unknown_count,
                            "unknown_frac": f"{unknown_frac:.6f}",
                            "assemblies": ";".join(chain_asmb_ids),
                            "npz_path": str(npz_path),
                            "token_module": token_module,
                        }
                    )
                    rows_written += 1

    print(f"Done. Wrote {rows_written} chains.")
    print(f"FASTA: {fasta_out}")
    print(f"META : {metadata_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Export chain-level protein sequences from parsed .npz files."
    )
    parser.add_argument(
        "--parsed_dir",
        type=Path,
        required=True,
        help="Directory containing parsed structure .npz files.",
    )
    parser.add_argument(
        "--fasta_out",
        type=Path,
        required=True,
        help="Output FASTA file path.",
    )
    parser.add_argument(
        "--metadata_out",
        type=Path,
        required=True,
        help="Output CSV metadata path.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="Minimum chain sequence length to keep.",
    )
    args = parser.parse_args()

    export_sequences(
        parsed_dir=args.parsed_dir,
        fasta_out=args.fasta_out,
        metadata_out=args.metadata_out,
        min_len=args.min_len,
    )


if __name__ == "__main__":
    main()
