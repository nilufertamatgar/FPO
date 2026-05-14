import csv
from html import parser
import json
import random
from pathlib import Path
from datetime import datetime
import argparse
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from torch.amp import autocast, GradScaler

from ema_pytorch import EMA

from data import all_atom_parse as aap
from data.all_atom_parse import load_structure_data, token_to_index
from data.residue_config import configure as configure_residues
from model.discrete_flow_aa import DiscreteFlow_AA
from model.zoidberg.zoidberg_GNN import Zoidberg_GNN


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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--max_ce_atoms", type=int, default=6000)
    parser.add_argument("--max_ce_residues", type=int, default=700)

    parser.add_argument(
        "--ce_train_csv",
        type=str,
        default="/home/proif/yyh/dataset/ADFLIP_train.csv",
    )

    parser.add_argument(
        "--train_parsed_root",
        type=str,
        default="/home/proif/nilufer/ADFLIP/dataset/pMT_parsed",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/proif/nilufer/ADFLIP/results/weights/ADFLIP_v1.pt",
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--samples_per_epoch", type=int, default=0)

    parser.add_argument(
    "--ce_parsed_root",
    type=str,
    default="/home/proif/yyh/dataset/train_parsed_subset",
)
    parser.add_argument("--ce_batch_size", type=int, default=1)


    parser.add_argument(
        "--train_csv_case11",
        type=str,
        default="/home/proif/nilufer/ADFLIP/pMT/Training_data/case11.csv",
    )
    parser.add_argument(
        "--train_csv_case12",
        type=str,
        default="/home/proif/nilufer/ADFLIP/pMT/Training_data/case12.csv",
    )
    parser.add_argument(
        "--train_csv_case2",
        type=str,
        default="/home/proif/nilufer/ADFLIP/pMT/Training_data/case2.csv",
    )
    parser.add_argument(
        "--train_csv_case3",
        type=str,
        default="/home/proif/nilufer/ADFLIP/pMT/Training_data/case3.csv",
    )

    parser.add_argument("--weight_case11", type=float, default=1.0)
    parser.add_argument("--weight_case12", type=float, default=4.0)
    parser.add_argument("--weight_case2", type=float, default=1.0)
    parser.add_argument("--weight_case3", type=float, default=1.0)


    return parser.parse_args()



RUN_NAME = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

LOG_DIR = Path("/home/proif/nilufer/ADFLIP/results/pMT/finetune_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"{RUN_NAME}.log"

CKPT_DIR = Path("/home/proif/nilufer/ADFLIP/results/pMT_finetune_checkpoints") / RUN_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def seq_to_tokens(seq: str) -> torch.Tensor:
    return torch.tensor(
        [AA1_TO_TOKEN.get(x, token_to_index["<UNK>"]) for x in seq.strip()],
        dtype=torch.long,
    )


def parsed_npz_from_input_path(input_path: str) -> Path:
    input_path = Path(input_path)
    pdb = input_path.name.split(".")[0]
    dataset_root = input_path.parent.parent
    split_name = input_path.parent.name
    out_root = dataset_root / f"{split_name}_parsed"
    subdir = pdb[1:3] if len(pdb) >= 3 else pdb[:2]
    return out_root / subdir / f"{pdb}.npz"

def process_ce_item(item, policy_model, device):
    base_data = structure_to_tensor_dict(item["structure"], device)
    pos_tokens = item["pos_tokens"].to(device, non_blocking=True)
    loss_mask = item["loss_mask"].to(device, non_blocking=True)

    L = pos_tokens.shape[0]
    time_t = random.uniform(0, 1)

    corruption_mask = torch.zeros(L, dtype=torch.bool, device=device)
    num_target = int(loss_mask.sum().item())
    corruption_mask[loss_mask] = torch.rand(num_target, device=device) < (1.0 - time_t)

    logits_pos, _, _ = forward_noisy(
        policy_model,
        base_data,
        pos_tokens,
        time_t,
        corruption_mask,
        add_coord_noise=False,
    )

    ce_loss = compute_ce_loss_from_logits(
        logits_pos,
        pos_tokens,
        loss_mask,
    )

    return {
        "seq_id": item["sequence_id"],
        "loss": ce_loss,
        "corrupt_count": int(corruption_mask.sum().item()),
    }


def pdb_to_npz(parsed_root: Path, pdb_id: str) -> Path:
    pdb_id = pdb_id.lower()
    return parsed_root / pdb_id[1:3] / f"{pdb_id}.npz"


def get_chain_internal_id(structure, chain_label: str) -> int:
    if hasattr(structure, "asym_id_to_chain_index"):
        mapping = getattr(structure, "asym_id_to_chain_index")

        if isinstance(mapping, np.ndarray) and mapping.dtype == object:
            mapping = mapping.item()

        if isinstance(mapping, dict) and chain_label in mapping:
            return int(mapping[chain_label])

    raise ValueError(f"Could not map chain label {chain_label} to internal chain_id")


class ChainCSVParsedStructureCEDataset(Dataset):
    def __init__(self, csv_path, parsed_root, max_atoms=6000, max_residues=700):
        self.csv_path = Path(csv_path)
        self.parsed_root = Path(parsed_root)
        self.max_atoms = max_atoms
        self.max_residues = max_residues
        self.rows = []
        self.skipped_large = []

        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                seq_id = row["Sequence_ID"].strip()
                gt_seq = row["GT_sequence"].strip()

                if "_" not in seq_id or not gt_seq:
                    continue

                pdb_id, chain_label = seq_id.split("_", 1)
                npz_path = pdb_to_npz(self.parsed_root, pdb_id)

                if not npz_path.exists():
                    continue

                try:
                    structure = load_structure_data(str(npz_path))

                    center_mask = np.asarray(structure.is_center).astype(bool)
                    protein_mask = np.asarray(structure.is_protein).astype(bool)

                    if hasattr(structure, "backbone_mask"):
                        backbone_mask = np.asarray(structure.backbone_mask).astype(bool)
                    else:
                        backbone_mask = np.ones_like(center_mask, dtype=bool)

                    if hasattr(structure, "not_pad_mask"):
                        not_pad_mask = np.asarray(structure.not_pad_mask).astype(bool)
                    else:
                        not_pad_mask = np.ones_like(center_mask, dtype=bool)

                    full_center_mask = center_mask & protein_mask & backbone_mask & not_pad_mask

                    num_atoms = len(structure.residue_token)
                    num_residues = int(full_center_mask.sum())

                    if num_atoms > self.max_atoms or num_residues > self.max_residues:
                        self.skipped_large.append(
                            {
                                "sequence_id": seq_id,
                                "npz_path": str(npz_path),
                                "num_atoms": num_atoms,
                                "num_residues": num_residues,
                            }
                        )
                        continue

                except Exception:
                    continue

                self.rows.append(
                    {
                        "sequence_id": seq_id,
                        "pdb_id": pdb_id,
                        "chain_label": chain_label,
                        "gt_seq": gt_seq,
                        "npz_path": npz_path,
                    }
                )

        if not self.rows:
            raise ValueError(f"No valid CE rows found from {self.csv_path}")

    def __len__(self):
        return len(self.rows)

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

        if hasattr(structure, "not_pad_mask"):
            not_pad_mask = np.asarray(structure.not_pad_mask).astype(bool)
        else:
            not_pad_mask = np.ones_like(center_mask, dtype=bool)

        full_center_mask = center_mask & protein_mask & backbone_mask & not_pad_mask
        target_center_mask = full_center_mask & chain_mask

        native_center_tokens = torch.tensor(
            np.asarray(structure.residue_token)[full_center_mask],
            dtype=torch.long,
        )

        loss_mask = torch.tensor(
            chain_mask[full_center_mask],
            dtype=torch.bool,
        )

        gt_tokens_chain = seq_to_tokens(row["gt_seq"])
        target_len = int(target_center_mask.sum())

        if len(gt_tokens_chain) != target_len:
            raise ValueError(
                f"CE GT length mismatch for {row['sequence_id']}: "
                f"{len(gt_tokens_chain)} vs {target_len}"
            )

        pos_tokens_full = native_center_tokens.clone()
        pos_tokens_full[loss_mask] = gt_tokens_chain

        return {
            "sequence_id": row["sequence_id"],
            "chain_label": row["chain_label"],
            "chain_internal_id": chain_internal_id,
            "structure": structure,
            "pos_tokens": pos_tokens_full,
            "loss_mask": loss_mask,
        }


class PMTMixedNegativeDataset(Dataset):
    def __init__(self, csv_case11, csv_case12, csv_case2, csv_case3, parsed_root, seed=1234):
        del parsed_root
        del seed

        self.csv_case11 = Path(csv_case11)
        self.csv_case12 = Path(csv_case12)
        self.csv_case2 = Path(csv_case2)
        self.csv_case3 = Path(csv_case3)

        self.input_structure_path = "/home/proif/nilufer/ADFLIP/dataset/pMT/pMT_ligand.pdb"
        self.npz_path = parsed_npz_from_input_path(self.input_structure_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Missing parsed structure file: {self.npz_path}")

        self.structure = load_structure_data(str(self.npz_path))

        center_mask = np.asarray(self.structure.is_center).astype(bool)
        protein_mask = np.asarray(self.structure.is_protein).astype(bool)
        if hasattr(self.structure, "backbone_mask"):
            backbone_mask = np.asarray(self.structure.backbone_mask).astype(bool)
        else:
            backbone_mask = np.ones_like(center_mask, dtype=bool)

        self.full_center_mask = center_mask & protein_mask & backbone_mask
        self.target_len = int(self.full_center_mask.sum())
        self.pocket_csv_path = "/dapustor/nilufer/ADFLIP/pMT/pocket_residues_modified.csv"
        self.pocket_position_indices = load_pocket_position_indices(self.pocket_csv_path)

        full_center_residue_indices = np.asarray(self.structure.residue_index)[self.full_center_mask]
        pocket_mask_np = np.isin(full_center_residue_indices, list(self.pocket_position_indices))
        self.pocket_mask = torch.tensor(pocket_mask_np, dtype=torch.bool)


        self.rows = []
        self.source_counts = {
            "case11": 0,
            "case12": 0,
            "case2": 0,
            "case3": 0,
        }

        self._load_pair_csv(self.csv_case11, neg_source="case11")
        self._load_pair_csv(self.csv_case12, neg_source="case12")
        self._load_pair_csv(self.csv_case2, neg_source="case2")
        self._load_pair_csv(self.csv_case3, neg_source="case3")

        if not self.rows:
            raise ValueError("No valid rows found in training CSVs")

        for row in self.rows:
            if len(row["pos_seq"]) != self.target_len:
                raise ValueError(
                    f"Positive length mismatch for {row['sequence_id']}: "
                    f"{len(row['pos_seq'])} vs {self.target_len}"
                )
            if len(row["neg_seq"]) != self.target_len:
                raise ValueError(
                    f"Negative length mismatch for {row['sequence_id']}: "
                    f"{len(row['neg_seq'])} vs {self.target_len}"
                )

    def _load_pair_csv(self, csv_path: Path, neg_source: str):
        if not csv_path.exists():
            return

        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                seq_id = row.get("Sequence_ID", "").strip()
                pos_seq = row.get("Positive_sequence", "").strip()
                neg_seq = row.get("Negative_sequence", "").strip()

                if not pos_seq or not neg_seq:
                    continue

                if not seq_id:
                    seq_id = f"{neg_source}_{idx}"

                unique_id = f"{neg_source}::{seq_id}::{idx}"

                self.rows.append(
                    {
                        "sequence_id": unique_id,
                        "base_sequence_id": seq_id,
                        "pos_seq": pos_seq,
                        "neg_seq": neg_seq,
                        "neg_source": neg_source,
                    }
                )
                self.source_counts[neg_source] += 1

    def get_sample_weights(self, source_weight_map):
        return [float(source_weight_map[row["neg_source"]]) for row in self.rows]

    def set_epoch(self, epoch: int):
        del epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        pos_tokens = seq_to_tokens(row["pos_seq"])
        neg_tokens = seq_to_tokens(row["neg_seq"])
        loss_mask = torch.ones_like(pos_tokens, dtype=torch.bool)

        return {
            "sequence_id": row["sequence_id"],
            "base_sequence_id": row["base_sequence_id"],
            "structure": self.structure,
            "loss_mask": loss_mask,
            "pocket_mask": self.pocket_mask.clone(),
            "pos_tokens": pos_tokens,
            "neg_tokens": neg_tokens,
            "neg_source": row["neg_source"],
        }




def add_coordinate_noise(base_data, backbone_std=0.5, other_std=2.0):
    noisy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_data.items()}

    if "position" not in noisy:
        raise KeyError("Expected 'position' in structure tensors")

    is_backbone = noisy["is_backbone"].bool()
    pos = noisy["position"]

    backbone_noise = torch.randn_like(pos) * backbone_std
    other_noise = torch.randn_like(pos) * other_std

    noise = torch.where(is_backbone.unsqueeze(-1), backbone_noise, other_noise)
    noisy["position"] = pos + noise
    return noisy


def forward_noisy(
    flow_model,
    base_data,
    target_tokens,
    time_t,
    corruption_mask,
    add_coord_noise=True,
):
    masked_input = target_tokens.clone()
    masked_input[corruption_mask] = aap.token_to_index["<MASK>"]

    if add_coord_noise:
        base_data_for_forward = add_coordinate_noise(
            base_data,
            backbone_std=0.5,
            other_std=2.0,
        )
    else:
        base_data_for_forward = base_data

    noisy_data = build_noisy_data(base_data_for_forward, masked_input, time_t)

    logits, _ = flow_model.model(
        noisy_data,
        torch.tensor([[time_t]], device=target_tokens.device),
    )

    log_probs = F.log_softmax(logits, dim=-1)
    return logits, log_probs, noisy_data



def compute_ce_loss_from_logits(logits, target_tokens, loss_mask):
    s_onehot = F.one_hot(target_tokens, num_classes=logits.size(-1)).float()
    s_onehot = s_onehot + 0.1 / float(s_onehot.size(-1))
    s_onehot = s_onehot / s_onehot.sum(-1, keepdim=True)

    log_probs = F.log_softmax(logits, dim=-1)
    per_pos_loss = -(s_onehot * log_probs).sum(-1)

    loss_mask = loss_mask.float()
    return (per_pos_loss * loss_mask).sum() / torch.clamp(loss_mask.sum(), min=1.0)

def load_pocket_position_indices(csv_path: str) -> set[int]:
    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read()

    matches = re.findall(r"[A-Z]{3}(\d+)", text)
    return {int(x) for x in matches}

def masked_sppo_loss(
    logp_pos,
    logp_neg,
    logp_ref_pos,
    logp_ref_neg,
    s_pos,
    s_neg,
    mask,
    pocket_mask,
    beta=0.2,
):
    diff_mask = ((s_pos != s_neg) | pocket_mask.bool()).float() * mask.float()
    mask_sum = torch.clamp(diff_mask.sum(dim=1), min=1.0)

    logp_pos_masked = (logp_pos * diff_mask).sum(dim=1) / mask_sum
    logp_neg_masked = (logp_neg * diff_mask).sum(dim=1) / mask_sum
    logp_ref_pos_masked = (logp_ref_pos * diff_mask).sum(dim=1) / mask_sum
    logp_ref_neg_masked = (logp_ref_neg * diff_mask).sum(dim=1) / mask_sum

    a = beta * (logp_pos_masked - logp_ref_pos_masked)
    b = beta * (logp_neg_masked - logp_ref_neg_masked)

    loss = (a - 0.5) ** 2 + (b + 0.5) ** 2
    return loss.mean()


def collate_pref(batch):
    return batch


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value


def build_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    configure_residues(include_nonstd_amino_acids=getattr(config.data, "include_nonstd_amino_acids", True))

    denoiser = Zoidberg_GNN(
        hidden_dim=config.zoidberg_denoiser.hidden_dim,
        encoder_hidden_dim=config.zoidberg_denoiser.hidden_dim,
        num_blocks=config.zoidberg_denoiser.num_layers,
        num_heads=config.zoidberg_denoiser.num_heads,
        k=config.zoidberg_denoiser.k_neighbors,
        num_positional_embeddings=config.zoidberg_denoiser.num_positional_embeddings,
        num_rbf=config.zoidberg_denoiser.num_rbf,
        augment_eps=config.zoidberg_denoiser.augment_eps,
        backbone_diheral=config.zoidberg_denoiser.backbone_diheral,
        dropout=config.zoidberg_denoiser.dropout,
        update_atom=config.zoidberg_denoiser.update_atom,
        num_decoder_blocks=config.zoidberg_denoiser.num_decoder_blocks,
        num_tfmr_heads=config.zoidberg_denoiser.num_tfmr_heads,
        num_tfmr_layers=config.zoidberg_denoiser.num_tfmr_layers,
        number_ligand_atom=config.zoidberg_denoiser.number_ligand_atom,
        mpnn_cutoff=config.zoidberg_denoiser.mpnn_cutoff,
        output_dim=config.zoidberg_denoiser.output_dim,
    )

    model = DiscreteFlow_AA(config, denoiser, min_t=0.0)
    model.load_state_dict(ckpt["model"])
    return model.to(device), config


def save_checkpoint(path, policy_model, ema, opt, scaler, config, global_step):
    payload = {
        "config": config,
        "step": global_step,
        "model": policy_model.state_dict(),
        "opt": opt.state_dict(),
        "ema": ema.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(payload, path)


def save_checkpoint_metadata(path, epoch, global_step, train_loss, val_loss, best_val_loss):
    meta = {
        "epoch": int(epoch),
        "step": int(global_step),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def structure_to_tensor_dict(structure, device):
    out = {}
    for k, v in structure.__dict__.items():
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                continue
            t = torch.from_numpy(v).to(device, non_blocking=True)
            if t.dtype == torch.float64:
                t = t.float()
            out[k] = t

    out["batch_index"] = torch.zeros_like(out["residue_index"], device=device)
    return out


def valid_residue_mask(data):
    mask = data["is_center"].bool() & data["is_protein"].bool()
    if "backbone_mask" in data:
        mask = mask & data["backbone_mask"].bool()
    if "not_pad_mask" in data:
        mask = mask & data["not_pad_mask"].bool()
    return mask


def check_backbone_consistency(base_data):
    protein_mask = base_data["is_protein"].bool()
    backbone_atom_mask = base_data["is_backbone"].bool()
    not_pad_mask = base_data["not_pad_mask"].bool()
    backbone_mask = base_data.get("backbone_mask", None)

    if backbone_mask is not None:
        backbone_mask = backbone_mask.bool()
    else:
        backbone_mask = torch.ones_like(backbone_atom_mask, dtype=torch.bool)

    center_mask = (
        base_data["is_center"].bool()
        & protein_mask
        & not_pad_mask
        & backbone_mask
    )
    overall_mask = (
        protein_mask
        & backbone_atom_mask
        & not_pad_mask
        & backbone_mask
    )

    n_centers = int(center_mask.sum().item())
    n_backbone_atoms = int(overall_mask.sum().item())
    ok = (n_backbone_atoms == 4 * n_centers)
    return ok, n_backbone_atoms, n_centers


def preflight_scan_dataset(dataset, device, log_fn, bad_ids_path):
    bad_rows = []

    log_fn("Starting preflight scan for backbone consistency")

    for idx in range(len(dataset)):
        item = dataset[idx]
        seq_id = item["sequence_id"]

        try:
            base_data = structure_to_tensor_dict(item["structure"], device)
            ok, n_backbone_atoms, n_centers = check_backbone_consistency(base_data)

            if not ok:
                bad_rows.append(
                    {
                        "index": idx,
                        "sequence_id": seq_id,
                        "backbone_atoms": n_backbone_atoms,
                        "centers": n_centers,
                        "expected_backbone_atoms": 4 * n_centers,
                        "error": "",
                    }
                )
        except Exception as e:
            bad_rows.append(
                {
                    "index": idx,
                    "sequence_id": seq_id,
                    "backbone_atoms": "",
                    "centers": "",
                    "expected_backbone_atoms": "",
                    "error": repr(e),
                }
            )

        if (idx + 1) % 500 == 0:
            log_fn(f"Preflight checked {idx + 1}/{len(dataset)} samples")

    with open(bad_ids_path, "w", encoding="utf-8") as f:
        f.write("index,sequence_id,backbone_atoms,centers,expected_backbone_atoms,error\n")
        for row in bad_rows:
            f.write(
                f"{row['index']},{row['sequence_id']},{row['backbone_atoms']},"
                f"{row['centers']},{row['expected_backbone_atoms']},{row['error']}\n"
            )

    log_fn(f"Preflight finished. Bad samples: {len(bad_rows)}")
    log_fn(f"Wrote bad sample report to: {bad_ids_path}")

    return {row["sequence_id"] for row in bad_rows}


def build_noisy_data(base_data, input_tokens, time_t):
    noisy_data = {k: v.clone() for k, v in base_data.items()}

    full_center_mask = valid_residue_mask(noisy_data)

    if int(full_center_mask.sum().item()) != int(input_tokens.shape[0]):
        raise ValueError(
            f"full-center/input mismatch: full_center={int(full_center_mask.sum().item())}, "
            f"input_tokens={int(input_tokens.shape[0])}"
        )

    noisy_data["residue_token"][full_center_mask] = input_tokens

    full_center_res_ids = noisy_data["residue_index"][full_center_mask]
    residue_to_compact = {
        int(res_id): i for i, res_id in enumerate(full_center_res_ids.tolist())
    }

    residue_is_masked = input_tokens == aap.token_to_index["<MASK>"]

    is_protein = noisy_data["is_protein"].bool()
    protein_res_ids = noisy_data["residue_index"][is_protein]

    atom_masked = torch.zeros_like(protein_res_ids, dtype=torch.bool)
    for i, res_id in enumerate(protein_res_ids.tolist()):
        compact_idx = residue_to_compact.get(int(res_id), None)
        if compact_idx is not None:
            atom_masked[i] = residue_is_masked[compact_idx]

    noisy_data["residue_token"][is_protein] = torch.where(
        atom_masked,
        torch.full_like(
            noisy_data["residue_token"][is_protein],
            aap.token_to_index["<MASK>"],
        ),
        noisy_data["residue_token"][is_protein],
    )

    keep_mask = torch.ones_like(noisy_data["residue_token"], dtype=torch.bool)
    sidechain_protein = (~noisy_data["is_backbone"].bool()) & is_protein

    side_res_ids = noisy_data["residue_index"][sidechain_protein]
    side_keep = torch.ones_like(side_res_ids, dtype=torch.bool)
    for i, res_id in enumerate(side_res_ids.tolist()):
        compact_idx = residue_to_compact.get(int(res_id), None)
        if compact_idx is not None:
            side_keep[i] = ~residue_is_masked[compact_idx]

    keep_mask[sidechain_protein] = side_keep

    out = {}
    skip_fields = {
        "noisy_residue_token",
        "interact_non_protein_res",
        "interact_ion_res",
        "interact_nucleotide_res",
        "interact_molecule_res",
        "is_mask",
    }

    for name, item in noisy_data.items():
        if name == "time_step":
            out[name] = torch.tensor([[time_t]], device=input_tokens.device)
        elif name not in skip_fields:
            out[name] = item[keep_mask].unsqueeze(0)

    return out


def gather_log_probs(flow_model, base_data, target_tokens, time_t, corruption_mask, loss_mask):
    logits, log_probs, noisy_data = forward_noisy(
        flow_model, base_data, target_tokens, time_t, corruption_mask
    )

    gathered = log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
    return gathered.unsqueeze(0), loss_mask.unsqueeze(0), logits, noisy_data


def process_single_item(item, policy_model, ref_model, device, beta, lambda_ce):
    seq_id = item["sequence_id"]

    base_data = structure_to_tensor_dict(item["structure"], device)
    pos_tokens = item["pos_tokens"].to(device, non_blocking=True)
    neg_tokens = item["neg_tokens"].to(device, non_blocking=True)
    loss_mask = item["loss_mask"].to(device, non_blocking=True)
    pocket_mask = item["pocket_mask"].to(device, non_blocking=True)

    L = pos_tokens.shape[0]
    time_t = random.uniform(0, 1)

    corruption_mask = torch.zeros(L, dtype=torch.bool, device=device)
    num_target = int(loss_mask.sum().item())
    corruption_mask[loss_mask] = torch.rand(num_target, device=device) < (1.0 - time_t)

    with autocast(device_type=device.type, enabled=False):
        logp_pos, mask, logits_pos, _ = gather_log_probs(
            policy_model, base_data, pos_tokens, time_t, corruption_mask, loss_mask
        )
        logp_neg, _, _, _ = gather_log_probs(
            policy_model, base_data, neg_tokens, time_t, corruption_mask, loss_mask
        )

        with torch.no_grad():
            logp_ref_pos, _, _, _ = gather_log_probs(
                ref_model, base_data, pos_tokens, time_t, corruption_mask, loss_mask
            )
            logp_ref_neg, _, _, _ = gather_log_probs(
                ref_model, base_data, neg_tokens, time_t, corruption_mask, loss_mask
            )

        sppo_loss = masked_sppo_loss(
            logp_pos,
            logp_neg,
            logp_ref_pos,
            logp_ref_neg,
            pos_tokens.unsqueeze(0),
            neg_tokens.unsqueeze(0),
            mask,
            pocket_mask.unsqueeze(0),
            beta=beta,
        )


        ce_loss = torch.zeros((), device=device)
        loss = sppo_loss


       

    diff_mask = (pos_tokens != neg_tokens) & loss_mask
    diff_count = int(diff_mask.sum().item())
    corrupt_count = int(corruption_mask.sum().item())

    diff_mask_f = diff_mask.unsqueeze(0).float()
    denom = torch.clamp(diff_mask_f.sum(dim=1), min=1.0)

    policy_pos_score = ((logp_pos * diff_mask_f).sum(dim=1) / denom).mean()
    policy_neg_score = ((logp_neg * diff_mask_f).sum(dim=1) / denom).mean()
    ref_pos_score = ((logp_ref_pos * diff_mask_f).sum(dim=1) / denom).mean()
    ref_neg_score = ((logp_ref_neg * diff_mask_f).sum(dim=1) / denom).mean()

    return {
        "seq_id": seq_id,
        "loss": loss,
        "sppo_loss": float(sppo_loss.detach().item()),
        "ce_loss": float(ce_loss.detach().item()),
        "diff_count": diff_count,
        "corrupt_count": corrupt_count,
        "neg_source": item["neg_source"],
        "policy_margin": float((policy_pos_score - policy_neg_score).detach().item()),
        "ref_margin": float((ref_pos_score - ref_neg_score).detach().item()),
        "pos_gain_vs_ref": float((policy_pos_score - ref_pos_score).detach().item()),
        "neg_gain_vs_ref": float((policy_neg_score - ref_neg_score).detach().item()),
    }


def main():
    args = parse_args()

    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    NUM_EPOCHS = args.num_epochs
    BETA = args.beta
    LOG_EVERY = args.log_every
    LAMBDA_CE = args.lambda_ce

    CE_PARSED_ROOT = args.ce_parsed_root
    CE_BATCH_SIZE = args.ce_batch_size
    CE_TRAIN_CSV = args.ce_train_csv
    MAX_CE_ATOMS = args.max_ce_atoms
    MAX_CE_RESIDUES = args.max_ce_residues

    TRAIN_CSV_CASE11 = args.train_csv_case11
    TRAIN_CSV_CASE12 = args.train_csv_case12
    TRAIN_CSV_CASE2 = args.train_csv_case2
    TRAIN_CSV_CASE3 = args.train_csv_case3

    TRAIN_PARSED_ROOT = args.train_parsed_root
    CKPT_PATH = args.ckpt
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    NUM_WORKERS = args.num_workers
    SEED = args.seed
    SAMPLES_PER_EPOCH = args.samples_per_epoch

    WEIGHT_CASE11 = args.weight_case11
    WEIGHT_CASE12 = args.weight_case12
    WEIGHT_CASE2 = args.weight_case2
    WEIGHT_CASE3 = args.weight_case3

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = False

        log(f"Using device: {device}")
        log(f"TRAIN_CSV_CASE11={TRAIN_CSV_CASE11}")
        log(f"TRAIN_CSV_CASE12={TRAIN_CSV_CASE12}")
        log(f"TRAIN_CSV_CASE2={TRAIN_CSV_CASE2}")
        log(f"TRAIN_CSV_CASE3={TRAIN_CSV_CASE3}")
        log(f"TRAIN_PARSED_ROOT={TRAIN_PARSED_ROOT}")
        log(f"CKPT_PATH={CKPT_PATH}")
        log(
            f"LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}, NUM_EPOCHS={NUM_EPOCHS}, "
            f"BETA={BETA}, LAMBDA_CE={LAMBDA_CE}"
        )
        log(
            f"BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}, "
            f"EFFECTIVE_BATCH={BATCH_SIZE * GRAD_ACCUM_STEPS}"
        )
        log(f"NUM_WORKERS={NUM_WORKERS}")
        log(
            f"SAMPLE_WEIGHTS="
            f"{{'case11': {WEIGHT_CASE11}, 'case12': {WEIGHT_CASE12}, "
            f"'case2': {WEIGHT_CASE2}, 'case3': {WEIGHT_CASE3}}}"
        )
        log(f"SAMPLES_PER_EPOCH={SAMPLES_PER_EPOCH}")
        log(f"CHECKPOINT_DIR={CKPT_DIR}")
        log(f"CE_TRAIN_CSV={CE_TRAIN_CSV}")
        log(f"CE_PARSED_ROOT={CE_PARSED_ROOT}")
        log(f"CE_BATCH_SIZE={CE_BATCH_SIZE}")
        log(f"MAX_CE_ATOMS={MAX_CE_ATOMS}")
        log(f"MAX_CE_RESIDUES={MAX_CE_RESIDUES}")

        policy_model, config = build_model_from_ckpt(CKPT_PATH, device)
        ref_model, _ = build_model_from_ckpt(CKPT_PATH, device)
        log("Loaded policy and reference models from checkpoint")

        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        opt = torch.optim.AdamW(
            policy_model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        ema = EMA(policy_model, beta=0.999, update_every=10)
        scaler = GradScaler(device="cuda", enabled=use_amp)

        train_ds = PMTMixedNegativeDataset(
            TRAIN_CSV_CASE11,
            TRAIN_CSV_CASE12,
            TRAIN_CSV_CASE2,
            TRAIN_CSV_CASE3,
            TRAIN_PARSED_ROOT,
            seed=SEED,
        )

        log(f"Initial mixed dataset size: {len(train_ds)}")
        log(f"Sample counts by source: {json.dumps(train_ds.source_counts, sort_keys=True)}")

        bad_ids_path = CKPT_DIR / "bad_train_samples_mixed.csv"
        bad_ids = preflight_scan_dataset(train_ds, device, log, bad_ids_path)

        if bad_ids:
            train_ds.rows = [
                row for row in train_ds.rows
                if row["sequence_id"] not in bad_ids
            ]

            filtered_counts = {"case11": 0, "case12": 0, "case2": 0, "case3": 0}
            for row in train_ds.rows:
                filtered_counts[row["neg_source"]] += 1
            train_ds.source_counts = filtered_counts

        log(f"Filtered mixed dataset size: {len(train_ds)}")
        log(f"Filtered sample counts by source: {json.dumps(train_ds.source_counts, sort_keys=True)}")

        source_weight_map = {
            "case11": WEIGHT_CASE11,
            "case12": WEIGHT_CASE12,
            "case2": WEIGHT_CASE2,
            "case3": WEIGHT_CASE3,
        }

        sample_weights = train_ds.get_sample_weights(source_weight_map)
        num_samples = SAMPLES_PER_EPOCH if SAMPLES_PER_EPOCH > 0 else len(sample_weights)

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
        )

        common_loader_kwargs = {
            "num_workers": NUM_WORKERS,
            "pin_memory": True,
            "collate_fn": collate_pref,
        }

        if NUM_WORKERS > 0:
            common_loader_kwargs["persistent_workers"] = True
            common_loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            **common_loader_kwargs,
        )

        ce_ds = ChainCSVParsedStructureCEDataset(
            CE_TRAIN_CSV,
            CE_PARSED_ROOT,
            max_atoms=MAX_CE_ATOMS,
            max_residues=MAX_CE_RESIDUES,
        )

        log(f"Initial CE dataset size: {len(ce_ds)}")
        log(f"Skipped large CE structures: {len(ce_ds.skipped_large)}")

        bad_ce_ids_path = CKPT_DIR / "bad_train_samples_ce.csv"
        bad_ce_ids = preflight_scan_dataset(ce_ds, device, log, bad_ce_ids_path)

        if bad_ce_ids:
            ce_ds.rows = [
                row for row in ce_ds.rows
                if row["sequence_id"] not in bad_ce_ids
            ]

        log(f"Filtered CE dataset size: {len(ce_ds)}")


        skipped_ce_path = CKPT_DIR / "skipped_large_ce_structures.csv"
        with open(skipped_ce_path, "w", encoding="utf-8") as f:
            f.write("sequence_id,npz_path,num_atoms,num_residues\n")
            for row in ce_ds.skipped_large:
                f.write(
                    f"{row['sequence_id']},{row['npz_path']},"
                    f"{row['num_atoms']},{row['num_residues']}\n"
                )

        log(f"Wrote skipped CE structure report to: {skipped_ce_path}")

        ce_loader = DataLoader(
            ce_ds,
            batch_size=CE_BATCH_SIZE,
            shuffle=True,
            **common_loader_kwargs,
        )

        global_step = 0
        best_train_loss = float("inf")

        for epoch in range(NUM_EPOCHS):
            train_ds.set_epoch(epoch)
            ce_iter = iter(ce_loader)

            policy_model.train()
            running = 0.0
            steps = 0
            accum_counter = 0

            opt.zero_grad(set_to_none=True)

            log(f"Starting epoch {epoch}")

            for batch in train_loader:
                try:
                    batch_total_loss = 0.0
                    batch_sppo_loss = 0.0
                    batch_ce_loss = 0.0
                    batch_diff_count = 0
                    batch_corrupt_count = 0
                    batch_seq_ids = []
                    batch_ce_seq_ids = []
                    batch_source_counts = {"case11": 0, "case12": 0, "case2": 0, "case3": 0}
                    batch_case_metrics = {}

                    for item in batch:
                        result = process_single_item(
                            item,
                            policy_model,
                            ref_model,
                            device,
                            BETA,
                            LAMBDA_CE,
                        )

                        batch_seq_ids.append(result["seq_id"])
                        batch_total_loss = batch_total_loss + result["loss"]
                        batch_sppo_loss += result["sppo_loss"]
                        batch_diff_count += result["diff_count"]
                        batch_corrupt_count += result["corrupt_count"]

                        src = result["neg_source"]

                        if src in batch_source_counts:
                            batch_source_counts[src] += 1
                        else:
                            batch_source_counts[src] = 1

                        if src not in batch_case_metrics:
                            batch_case_metrics[src] = {
                                "count": 0,
                                "policy_margin": 0.0,
                                "ref_margin": 0.0,
                                "pos_gain_vs_ref": 0.0,
                                "neg_gain_vs_ref": 0.0,
                            }

                        batch_case_metrics[src]["count"] += 1
                        batch_case_metrics[src]["policy_margin"] += result["policy_margin"]
                        batch_case_metrics[src]["ref_margin"] += result["ref_margin"]
                        batch_case_metrics[src]["pos_gain_vs_ref"] += result["pos_gain_vs_ref"]
                        batch_case_metrics[src]["neg_gain_vs_ref"] += result["neg_gain_vs_ref"]

                    sppo_loss = batch_total_loss / len(batch)

                    ce_loss = torch.zeros((), device=device)
                    ce_corrupt_count = 0

                    try:
                        try:
                            ce_batch = next(ce_iter)
                        except StopIteration:
                            ce_iter = iter(ce_loader)
                            ce_batch = next(ce_iter)

                        ce_total_loss = 0.0

                        for ce_item in ce_batch:
                            ce_result = process_ce_item(
                                ce_item,
                                policy_model,
                                device,
                            )

                            ce_total_loss = ce_total_loss + ce_result["loss"]
                            ce_corrupt_count += ce_result["corrupt_count"]
                            batch_ce_seq_ids.append(ce_result["seq_id"])

                        ce_loss = ce_total_loss / len(ce_batch)

                    except torch.cuda.OutOfMemoryError as e:
                        ce_ids = []
                        if "ce_batch" in locals():
                            ce_ids = [
                                x.get("sequence_id", "unknown")
                                for x in ce_batch
                            ]

                        log(
                            f"CE OOM, skipping CE batch: "
                            f"ce_seq_ids={ce_ids}, error={repr(e)}"
                        )

                        torch.cuda.empty_cache()
                        ce_loss = torch.zeros((), device=device)
                        ce_corrupt_count = 0
                        batch_ce_seq_ids = []

                    loss = sppo_loss + (LAMBDA_CE * ce_loss)
                    loss_for_backward = loss / GRAD_ACCUM_STEPS

                    batch_ce_loss = float(ce_loss.detach().item())
                    batch_corrupt_count += ce_corrupt_count

                    scaler.scale(loss_for_backward).backward()
                    accum_counter += 1

                    should_step = accum_counter == GRAD_ACCUM_STEPS
                    is_last_batch = (steps + 1) == len(train_loader)

                    if should_step or is_last_batch:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                        ema.update()
                        global_step += 1
                        accum_counter = 0

                    running += float(loss.detach().item())
                    steps += 1

                    if steps % LOG_EVERY == 0:
                        avg_so_far = running / max(steps, 1)

                        case_metric_strs = []
                        for case_name, stats in sorted(batch_case_metrics.items()):
                            n = max(stats["count"], 1)
                            case_metric_strs.append(
                                f"{case_name}:n={stats['count']},"
                                f"pm={stats['policy_margin'] / n:.4f},"
                                f"rm={stats['ref_margin'] / n:.4f},"
                                f"pg={stats['pos_gain_vs_ref'] / n:.4f},"
                                f"ng={stats['neg_gain_vs_ref'] / n:.4f}"
                            )

                        log(
                            f"epoch={epoch} batch={steps} opt_step={global_step} "
                            f"batch_size={len(batch)} "
                            f"source_counts={batch_source_counts} "
                            f"seq_ids={batch_seq_ids} "
                            f"ce_seq_ids={batch_ce_seq_ids} "
                            f"avg_sppo_loss={batch_sppo_loss / len(batch):.6f} "
                            f"ce_loss={batch_ce_loss:.6f} "
                            f"total_loss={loss.item():.6f} "
                            f"avg_epoch_loss={avg_so_far:.6f} "
                            f"diff_count={batch_diff_count} "
                            f"corrupt_count={batch_corrupt_count} "
                            f"case_metrics={' | '.join(case_metric_strs)}"
                        )

                except torch.cuda.OutOfMemoryError as e:
                    log(f"OOM for batch starting with {batch[0]['sequence_id']}: {repr(e)}")
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    accum_counter = 0
                    continue

                except Exception as e:
                    log(f"Batch failed for {batch[0]['sequence_id']}: {repr(e)}")
                    raise

            train_loss = running / max(steps, 1)
            log(f"Finished epoch {epoch} train_loss={train_loss:.6f}")

            epoch_ckpt_path = CKPT_DIR / f"epoch_{epoch}.pt"
            epoch_meta_path = CKPT_DIR / f"epoch_{epoch}.json"

            save_checkpoint(
                epoch_ckpt_path,
                policy_model,
                ema,
                opt,
                scaler,
                config,
                global_step,
            )

            save_checkpoint_metadata(
                epoch_meta_path,
                epoch,
                global_step,
                train_loss,
                train_loss,
                best_train_loss,
            )

            log(f"Saved epoch checkpoint: {epoch_ckpt_path}")
            log(f"Saved epoch metadata: {epoch_meta_path}")

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_ckpt_path = CKPT_DIR / "best.pt"
                best_meta_path = CKPT_DIR / "best.json"

                save_checkpoint(
                    best_ckpt_path,
                    policy_model,
                    ema,
                    opt,
                    scaler,
                    config,
                    global_step,
                )

                save_checkpoint_metadata(
                    best_meta_path,
                    epoch,
                    global_step,
                    train_loss,
                    train_loss,
                    best_train_loss,
                )

                log(
                    f"Saved new best checkpoint: {best_ckpt_path} "
                    f"(best_train_loss={best_train_loss:.6f})"
                )
                log(f"Saved new best metadata: {best_meta_path}")

        log("Finetuning complete")

    except Exception as e:
        log(f"ERROR: {repr(e)}")
        raise

if __name__ == "__main__":
    main()