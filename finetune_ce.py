import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ema_pytorch import EMA

from data import all_atom_parse as aap
from data.residue_config import configure as configure_residues
from model.discrete_flow_aa import DiscreteFlow_AA
from model.zoidberg.zoidberg_GNN import Zoidberg_GNN
from preference_dataset import PreferenceDataset


TRAIN_CSV = "/dapustor/nilufer/ADFLIP/finetuning_data/csvs/adflip_train_strict_1.csv"
VALID_CSV = "/dapustor/nilufer/ADFLIP/finetuning_data/csvs/adflip_valid_strict_1.csv"

TRAIN_PARSED_ROOT = "/dapustor/nilufer/ADFLIP/dataset/train_parsed"
VALID_PARSED_ROOT = "/dapustor/nilufer/ADFLIP/dataset/valid_parsed"

CKPT_PATH = "/dapustor/nilufer/ADFLIP/results/weights/ADFLIP_v1.pt"

LR = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
BETA = 0.2
LOG_EVERY = 100
LAMBDA_CE = 1.0


RUN_NAME = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

LOG_DIR = Path("/dapustor/nilufer/ADFLIP/results/finetune_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"{RUN_NAME}.log"

CKPT_DIR = Path("/dapustor/nilufer/ADFLIP/results/finetune_checkpoints") / RUN_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        

def forward_noisy(flow_model, base_data, target_tokens, time_t, corruption_mask):
    masked_input = target_tokens.clone()
    masked_input[corruption_mask] = aap.token_to_index["<MASK>"]

    noisy_data = build_noisy_data(base_data, masked_input, time_t)
    logits, _ = flow_model.model(
        noisy_data,
        torch.tensor([[time_t]], device=target_tokens.device),
    )
    log_probs = F.log_softmax(logits, dim=-1)
    return logits, log_probs, noisy_data


def compute_ce_loss_from_logits(flow_model, logits, noisy_data, target_tokens):
    center_mask = (
        noisy_data["is_center"].bool()
        & noisy_data["is_protein"].bool()
        & noisy_data["backbone_mask"].bool()
        & noisy_data["not_pad_mask"].bool()
    ).squeeze(0)

    target = noisy_data["residue_token"].squeeze(0)[center_mask].clone()
    target[:] = target_tokens

    
    s_onehot = F.one_hot(target, num_classes=logits.size(-1)).float()
    s_onehot = s_onehot + 0.1 / float(s_onehot.size(-1))
    s_onehot = s_onehot / s_onehot.sum(-1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(s_onehot * log_probs).sum(-1)
    return loss.mean()

    




def masked_sppo_loss(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, s_pos, s_neg, mask, beta=0.2):
    diff_mask = (s_pos != s_neg).float() * mask.float()
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
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


def save_checkpoint(path, policy_model, ema, opt, config, global_step):
    payload = {
        "config": config,
        "step": global_step,
        "model": policy_model.state_dict(),
        "opt": opt.state_dict(),
        "ema": ema.state_dict(),
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
            t = torch.from_numpy(v).to(device)
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
    center_mask = valid_residue_mask(noisy_data)

    if int(center_mask.sum().item()) != int(input_tokens.shape[0]):
        raise ValueError(
            f"designable/input mismatch: designable={int(center_mask.sum().item())}, "
            f"input_tokens={int(input_tokens.shape[0])}"
        )

    noisy_data["residue_token"][center_mask] = input_tokens

    # Map full residue ids to  compact designable positions
    center_res_ids = noisy_data["residue_index"][center_mask]
    residue_to_compact = {
        int(res_id): i for i, res_id in enumerate(center_res_ids.tolist())
    }

    residue_is_masked = (input_tokens == aap.token_to_index["<MASK>"])

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



def gather_log_probs(flow_model, base_data, target_tokens, time_t, corruption_mask):
    logits, log_probs, noisy_data = forward_noisy(
        flow_model, base_data, target_tokens, time_t, corruption_mask
    )

    gathered = log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
    score_mask = torch.ones_like(
        target_tokens, dtype=torch.bool, device=target_tokens.device
    )
    return gathered.unsqueeze(0), score_mask.unsqueeze(0), logits, noisy_data



def evaluate(policy_model, ref_model, loader, device, beta):
    policy_model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            item = batch[0]
            seq_id = item["sequence_id"]
            base_data = structure_to_tensor_dict(item["structure"], device)
            pos_tokens = item["pos_tokens"].to(device)
            neg_tokens = item["neg_tokens"].to(device)

            L = pos_tokens.shape[0]
            time_t = random.uniform(0, 1)
            corruption_mask = torch.rand(L, device=device) < (1.0 - time_t)

            logp_pos, mask, logits_pos, noisy_pos = gather_log_probs(
                policy_model, base_data, pos_tokens, time_t, corruption_mask
            )
            logp_neg, _, _, _ = gather_log_probs(
                policy_model, base_data, neg_tokens, time_t, corruption_mask
            )
            logp_ref_pos, _, _, _ = gather_log_probs(
                ref_model, base_data, pos_tokens, time_t, corruption_mask
            )
            logp_ref_neg, _, _, _ = gather_log_probs(
                ref_model, base_data, neg_tokens, time_t, corruption_mask
            )

            sppo_loss = masked_sppo_loss(
                logp_pos,
                logp_neg,
                logp_ref_pos,
                logp_ref_neg,
                pos_tokens.unsqueeze(0),
                neg_tokens.unsqueeze(0),
                mask,
                beta=beta,
            )

            ce_loss = compute_ce_loss_from_logits(policy_model, logits_pos, noisy_pos, pos_tokens)
            loss = sppo_loss + LAMBDA_CE * ce_loss

            total += loss.item()
            count += 1

    return total / max(count, 1)


def main():
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")
        log(f"TRAIN_CSV={TRAIN_CSV}")
        log(f"VALID_CSV={VALID_CSV}")
        log(f"TRAIN_PARSED_ROOT={TRAIN_PARSED_ROOT}")
        log(f"VALID_PARSED_ROOT={VALID_PARSED_ROOT}")
        log(f"CKPT_PATH={CKPT_PATH}")
        log(f"LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}, NUM_EPOCHS={NUM_EPOCHS}, BETA={BETA}")
        log(f"CHECKPOINT_DIR={CKPT_DIR}")

        policy_model, config = build_model_from_ckpt(CKPT_PATH, device)
        ref_model, _ = build_model_from_ckpt(CKPT_PATH, device)
        log("Loaded policy and reference models from checkpoint")

        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        opt = torch.optim.AdamW(policy_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        ema = EMA(policy_model, beta=0.999, update_every=10)

        train_ds = PreferenceDataset(TRAIN_CSV, TRAIN_PARSED_ROOT)
        valid_ds = PreferenceDataset(VALID_CSV, VALID_PARSED_ROOT)

        log(f"Initial train dataset size: {len(train_ds)}")
        log(f"Initial valid dataset size: {len(valid_ds)}")

        bad_train_path = CKPT_DIR / "bad_train_samples.csv"
        bad_valid_path = CKPT_DIR / "bad_valid_samples.csv"

        bad_train_ids = preflight_scan_dataset(train_ds, device, log, bad_train_path)
        bad_valid_ids = preflight_scan_dataset(valid_ds, device, log, bad_valid_path)

        if bad_train_ids:
            train_ds.rows = [row for row in train_ds.rows if row["sequence_id"] not in bad_train_ids]
        if bad_valid_ids:
            valid_ds.rows = [row for row in valid_ds.rows if row["sequence_id"] not in bad_valid_ids]

        log(f"Filtered train dataset size: {len(train_ds)}")
        log(f"Filtered valid dataset size: {len(valid_ds)}")

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_pref)
        valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, collate_fn=collate_pref)

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(NUM_EPOCHS):
            train_ds.set_epoch(epoch)
            valid_ds.set_epoch(epoch)

            policy_model.train()
            running = 0.0
            steps = 0

            log(f"Starting epoch {epoch}")

            for batch in train_loader:
                item = batch[0]
                seq_id = item["sequence_id"]

                try:
                    base_data = structure_to_tensor_dict(item["structure"], device)
                    pos_tokens = item["pos_tokens"].to(device)
                    neg_tokens = item["neg_tokens"].to(device)

                    L = pos_tokens.shape[0]
                    time_t = random.uniform(0, 1)
                    corruption_mask = torch.rand(L, device=device) < (1.0 - time_t)


                    logp_pos, mask, logits_pos, noisy_pos = gather_log_probs(
                    policy_model, base_data, pos_tokens, time_t, corruption_mask
                    )
                    logp_neg, _, _, _ = gather_log_probs(
                        policy_model, base_data, neg_tokens, time_t, corruption_mask
                    )

                    with torch.no_grad():
                        logp_ref_pos, _, _, _ = gather_log_probs(
                            ref_model, base_data, pos_tokens, time_t, corruption_mask
                        )
                        logp_ref_neg, _, _, _ = gather_log_probs(
                            ref_model, base_data, neg_tokens, time_t, corruption_mask
                        )


                    sppo_loss = masked_sppo_loss(
                        logp_pos,
                        logp_neg,
                        logp_ref_pos,
                        logp_ref_neg,
                        pos_tokens.unsqueeze(0),
                        neg_tokens.unsqueeze(0),
                        mask,
                        beta=BETA,
                    )

                    ce_loss = compute_ce_loss_from_logits(policy_model, logits_pos, noisy_pos, pos_tokens)
                    loss = sppo_loss + LAMBDA_CE * ce_loss


                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    opt.step()
                    ema.update()

                    running += loss.item()
                    steps += 1
                    global_step += 1

                    if global_step % LOG_EVERY == 0:
                        avg_so_far = running / max(steps, 1)
                        diff_count = int((pos_tokens != neg_tokens).sum().item())
                        log(
                            f"epoch={epoch} step={global_step} seq_id={seq_id} "
                            f"batch_loss={loss.item():.6f} avg_epoch_loss={avg_so_far:.6f} "
                            f"L={L} diff_count={diff_count} corrupt_count={int(corruption_mask.sum().item())}"
                        )

                except Exception as e:
                    log(f"Batch failed for sequence_id={seq_id}: {repr(e)}")
                    raise

            train_loss = running / max(steps, 1)
            val_loss = evaluate(policy_model, ref_model, valid_loader, device, BETA)

            log(f"Finished epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            epoch_ckpt_path = CKPT_DIR / f"epoch_{epoch}.pt"
            epoch_meta_path = CKPT_DIR / f"epoch_{epoch}.json"

            save_checkpoint(
                epoch_ckpt_path,
                policy_model,
                ema,
                opt,
                config,
                global_step,
            )
            save_checkpoint_metadata(
                epoch_meta_path,
                epoch,
                global_step,
                train_loss,
                val_loss,
                best_val_loss,
            )
            log(f"Saved epoch checkpoint: {epoch_ckpt_path}")
            log(f"Saved epoch metadata: {epoch_meta_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = CKPT_DIR / "best.pt"
                best_meta_path = CKPT_DIR / "best.json"

                save_checkpoint(
                    best_ckpt_path,
                    policy_model,
                    ema,
                    opt,
                    config,
                    global_step,
                )
                save_checkpoint_metadata(
                    best_meta_path,
                    epoch,
                    global_step,
                    train_loss,
                    val_loss,
                    best_val_loss,
                )
                log(f"Saved new best checkpoint: {best_ckpt_path} (best_val_loss={best_val_loss:.6f})")
                log(f"Saved new best metadata: {best_meta_path}")

        log("Finetuning complete")

    except Exception as e:
        log(f"ERROR: {repr(e)}")
        raise


if __name__ == "__main__":
    main()
