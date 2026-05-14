import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.all_atom_parse import load_structure_data, token_to_index


AA1_TO_TOKEN = {
    "A": token_to_index["ALA"],
    "R": token_to_index["ARG"],
    "N": token_to_index["ASN"],
    "D": token_to_index["ASP"],
    "C": token_to_index["CYS"],
    "Q": token_to_index["GLN"],
    "E": token_to_index["GLU"],
    "G": token_to_index["GLY"],
    "H": token_to_index["HIS"],
    "I": token_to_index["ILE"],
    "L": token_to_index["LEU"],
    "K": token_to_index["LYS"],
    "M": token_to_index["MET"],
    "F": token_to_index["PHE"],
    "P": token_to_index["PRO"],
    "S": token_to_index["SER"],
    "T": token_to_index["THR"],
    "W": token_to_index["TRP"],
    "Y": token_to_index["TYR"],
    "V": token_to_index["VAL"],
    "X": token_to_index["<UNK>"],
}


def seq_to_tokens(seq: str) -> torch.Tensor:
    return torch.tensor(
        [AA1_TO_TOKEN.get(x, token_to_index["<UNK>"]) for x in seq],
        dtype=torch.long,
    )


def pdb_to_npz(parsed_root: Path, pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    return parsed_root / pdb_id[1:3] / f"{pdb_id}.npz"


def get_chain_internal_id(structure, chain_label: str) -> int:
    if hasattr(structure, "asym_id_to_chain_index"):
        mapping = getattr(structure, "asym_id_to_chain_index")
        if isinstance(mapping, dict) and chain_label in mapping:
            return int(mapping[chain_label])

    raise ValueError(f"Could not map chain label {chain_label} to internal chain_id")


class PreferenceDataset(Dataset):
    def __init__(self, csv_path, parsed_root):
        self.csv_path = Path(csv_path)
        self.parsed_root = Path(parsed_root)
        self.rows = []
        self.current_epoch = 0

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_id = row["Sequence_ID"].strip()
                gt_seq = row["GT_sequence"].strip()
                if "_" not in seq_id:
                    continue

                pdb_id, chain_label = seq_id.split("_", 1)
                npz_path = pdb_to_npz(self.parsed_root, pdb_id)
                if not npz_path.exists():
                    continue

                negatives = []
                for k, v in row.items():
                    if k.startswith("Neg_seq_") and v and v.strip():
                        negatives.append(v.strip())

                if not negatives:
                    continue

                self.rows.append(
                    {
                        "sequence_id": seq_id,
                        "pdb_id": pdb_id,
                        "chain_label": chain_label,
                        "gt_seq": gt_seq,
                        "negatives": negatives,
                        "npz_path": npz_path,
                    }
                )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self):
        return len(self.rows)

    def _pick_negative(self, negatives):
        if self.current_epoch < len(negatives):
            return negatives[self.current_epoch]
        return random.choice(negatives)

    def __getitem__(self, idx):
        row = self.rows[idx]
        structure = load_structure_data(str(row["npz_path"]))
        chain_internal_id = get_chain_internal_id(structure, row["chain_label"])

        center_mask = np.asarray(structure.is_center).astype(bool)
        protein_mask = np.asarray(structure.is_protein).astype(bool)
        chain_mask = np.asarray(structure.chain_id) == chain_internal_id

        if hasattr(structure, "backbone_mask"):
            backbone_mask = np.asarray(structure.backbone_mask).astype(bool)
        else:
            backbone_mask = np.ones_like(center_mask, dtype=bool)

        full_center_mask = center_mask & protein_mask & backbone_mask
        target_center_mask = full_center_mask & chain_mask

        native_center_tokens = torch.tensor(
            np.asarray(structure.residue_token)[full_center_mask],
            dtype=torch.long,
        )

        loss_mask = torch.tensor(
            chain_mask[full_center_mask],
            dtype=torch.bool,
        )

        pos_tokens_chain = seq_to_tokens(row["gt_seq"])
        neg_seq = self._pick_negative(row["negatives"])
        neg_tokens_chain = seq_to_tokens(neg_seq)

        target_len = int(target_center_mask.sum())
        if len(pos_tokens_chain) != target_len:
            raise ValueError(
                f"Positive length mismatch for {row['sequence_id']}: "
                f"{len(pos_tokens_chain)} vs {target_len}"
            )
        if len(neg_tokens_chain) != target_len:
            raise ValueError(
                f"Negative length mismatch for {row['sequence_id']}: "
                f"{len(neg_tokens_chain)} vs {target_len}"
            )

        pos_tokens_full = native_center_tokens.clone()
        neg_tokens_full = native_center_tokens.clone()

        pos_tokens_full[loss_mask] = pos_tokens_chain
        neg_tokens_full[loss_mask] = neg_tokens_chain

        return {
            "sequence_id": row["sequence_id"],
            "chain_label": row["chain_label"],
            "chain_internal_id": chain_internal_id,
            "structure": structure,
            "loss_mask": loss_mask,
            "pos_tokens": pos_tokens_full,
            "neg_tokens": neg_tokens_full,
        }
