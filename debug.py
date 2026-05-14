import random
import numpy as np
import torch
import torch.nn.functional as F

from data import all_atom_parse as aap
from data.residue_config import configure as configure_residues
from model.discrete_flow_aa import DiscreteFlow_AA
from model.zoidberg.zoidberg_GNN import Zoidberg_GNN
from preference_dataset import PreferenceDataset


TRAIN_CSV = "/dapustor/nilufer/ADFLIP/finetuning_data/adflip_train_strict.csv"
TRAIN_PARSED_ROOT = "/dapustor/nilufer/ADFLIP/dataset/train_parsed"
CKPT_PATH = "/dapustor/nilufer/ADFLIP/results/weights/ADFLIP_v1.pt"

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


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
    return model.to(device)


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
    return data["is_center"].bool() & data["is_protein"].bool()


def build_noisy_data(base_data, input_tokens, time_t):
    noisy_data = {k: v.clone() for k, v in base_data.items()}
    center_mask = valid_residue_mask(noisy_data)

    noisy_data["residue_token"][center_mask] = input_tokens

    residue_is_masked = (input_tokens == aap.token_to_index["<MASK>"])
    is_protein = noisy_data["is_protein"].bool()
    res_idx_protein = noisy_data["residue_index"][is_protein]
    atom_masked = residue_is_masked[res_idx_protein]

    noisy_data["residue_token"][is_protein] = torch.where(
        atom_masked,
        torch.tensor(aap.token_to_index["<MASK>"], device=input_tokens.device),
        noisy_data["residue_token"][is_protein],
    )

    keep_mask = torch.ones_like(noisy_data["residue_token"], dtype=torch.bool)
    sidechain_protein = ~noisy_data["is_backbone"].bool() & is_protein
    keep_mask[sidechain_protein] = ~residue_is_masked[noisy_data["residue_index"][sidechain_protein]]

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
    masked_input = target_tokens.clone()
    masked_input[corruption_mask] = aap.token_to_index["<MASK>"]

    noisy_data = build_noisy_data(base_data, masked_input, time_t)
    logits, _ = flow_model.model(noisy_data, torch.tensor([[time_t]], device=target_tokens.device))

    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)

    score_mask = torch.ones_like(target_tokens, dtype=torch.bool, device=target_tokens.device)
    return logits, gathered.unsqueeze(0), score_mask.unsqueeze(0)


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds = PreferenceDataset(TRAIN_CSV, TRAIN_PARSED_ROOT)
item = ds[0]

print("sequence_id:", item["sequence_id"])

base_data = structure_to_tensor_dict(item["structure"], device)
pos_tokens = item["pos_tokens"].to(device)
neg_tokens = item["neg_tokens"].to(device)

print("pos_tokens shape:", pos_tokens.shape)
print("neg_tokens shape:", neg_tokens.shape)

L = pos_tokens.shape[0]
time_t = random.uniform(0.05, 0.95)
corruption_mask = torch.rand(L, device=device) < time_t

print("time_t:", time_t)
print("num corrupted residues:", int(corruption_mask.sum().item()))
print("num differing residues:", int((pos_tokens != neg_tokens).sum().item()))

policy_model = build_model_from_ckpt(CKPT_PATH, device)
ref_model = build_model_from_ckpt(CKPT_PATH, device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

policy_model.eval()

logits_pos, logp_pos, mask = gather_log_probs(policy_model, base_data, pos_tokens, time_t, corruption_mask)
_, logp_neg, _ = gather_log_probs(policy_model, base_data, neg_tokens, time_t, corruption_mask)

with torch.no_grad():
    _, logp_ref_pos, _ = gather_log_probs(ref_model, base_data, pos_tokens, time_t, corruption_mask)
    _, logp_ref_neg, _ = gather_log_probs(ref_model, base_data, neg_tokens, time_t, corruption_mask)

loss = masked_sppo_loss(
    logp_pos,
    logp_neg,
    logp_ref_pos,
    logp_ref_neg,
    pos_tokens.unsqueeze(0),
    neg_tokens.unsqueeze(0),
    mask,
    beta=0.2,
)

print("logits_pos shape:", logits_pos.shape)
print("logp_pos shape:", logp_pos.shape)
print("logp_neg shape:", logp_neg.shape)
print("mask shape:", mask.shape)
print("loss:", float(loss.item()))
print("loss finite:", torch.isfinite(loss).item())
