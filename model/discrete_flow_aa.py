import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
import matplotlib.pyplot as plt
from data.all_atom_parse import residue_tokens, pdb2data,restype_1to3,make_merged_pdb,cif_to_pdb
import os
from PIPPack.model.modules import PIPPackFineTune
from PIPPack.inference import replace_protein_sequence, pdbs_from_prediction
from PIPPack.data.protein import from_pdb_file
from PIPPack.data.top2018_dataset import transform_structure, collate_fn
from PIPPack.ensembled_inference import sample_epoch
import pickle
import time
from typing import List

def _is_abnormal_pdb_line(line: str) -> bool:
    """Return *True* if the ATOM / HETATM / ANISOU record is shifted left."""

    if not line.startswith(("ATOM  ", "HETATM", "ANISOU")):
        return False

    # 1. residue name length > 3 (columns 18‑20 plus possible spill‑over)
    res_field = line[17:21]  # cols 18‑21 (0‑based slice 17:21)
    if len(res_field.strip()) > 3:
        return True

    # 2. column 21 (index 20) must be blank in a valid PDB
    if line[20] != " ":
        return True

    # 3. column 23 (index 22) is the first of the 4‑char residue number field
    #    If it contains a minus sign, the record is shifted left.
    if line[22] == "-":
        return True

    return False


# ---------------------------------------------------------------------------
# Helper: clean PDB file in place
# ---------------------------------------------------------------------------

def _clean_pdb_file(pdb_file: str) -> None:
    with open(pdb_file, "r", encoding="utf 8") as fh:
        good_lines: List[str] = [ln for ln in fh if not _is_abnormal_pdb_line(ln)]
    with open(pdb_file, "w", encoding="utf 8") as fh:
        fh.writelines(good_lines)


class DiscreteFlow_AA(nn.Module):

    def __init__(self, config, model, min_t=0.0,sidechain_packing=False,sample_save_path='results/sample_seq/',num_sc_models = 1, **kwargs):
        super().__init__()
        self.config = config
        self.model = model
        self.min_t = min_t
        self.sample_save_path = sample_save_path
        self.label_smoothing = config.training.label_smoothing
        if sidechain_packing:
            self.sc,self.infer_cfg = self.load_sc_model(device = next(self.model.parameters()).device,num_models = num_sc_models)


    def load_sc_model(self,device,num_models=3):
        model_names = ["pippack_model_1", "pippack_model_2", "pippack_model_3"]
        models = []
        inference_cfg = '/dapustor/nilufer/ADFLIP/PIPPack/model_weights/inference.pickle'
        with open(inference_cfg, 'rb') as f:
            infer_cfg = pickle.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model_name in model_names[:num_models]:
            cfg_file = f'/dapustor/nilufer/ADFLIP/PIPPack/model_weights/{model_name}_config.pickle'
            ckpt_file = f'/dapustor/nilufer/ADFLIP/PIPPack/model_weights/{model_name}_ckpt.pt'

            with open(cfg_file, 'rb') as f:
                cfg = pickle.load(f)
            
            model = PIPPackFineTune(
                node_features = cfg.model.node_features,
                edge_features = cfg.model.edge_features,
                hidden_dim = cfg.model.hidden_dim,
                num_mpnn_layers = cfg.model.num_mpnn_layers,
                k_neighbors = cfg.model.k_neighbors,
                augment_eps = cfg.model.augment_eps,
                use_ipmp = cfg.model.use_ipmp,
                use_ipmp_ipa = cfg.model.use_ipmp_ipa,
                n_points = cfg.model.n_points,
                dropout = cfg.model.dropout,
                act = cfg.model.act,
                predict_bin_chi = cfg.model.predict_bin_chi,
                n_chi_bins = cfg.model.n_chi_bins,
                predict_offset = cfg.model.predict_offset,
                position_scale = cfg.model.position_scale,
                recycle_strategy = cfg.model.recycle_strategy,
                recycle_SC_D_sc = cfg.model.recycle_SC_D_sc,
                recycle_SC_D_probs = cfg.model.recycle_SC_D_probs,
                recycle_X = cfg.model.recycle_X,
                mask_distances = cfg.model.mask_distances,
                loss = cfg.model.loss,
            )
            state_dicts = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(state_dicts['model_state_dict'])
            model = model.to(device)
            models.append(model)
        return models,infer_cfg

    def compute_loss(self, logits, data):
        if self.label_smoothing:
            center_mask = data["is_center"] & data["is_protein"]
            if "backbone_mask" in data:
                center_mask = center_mask & data["backbone_mask"]
            target = data["residue_token"][center_mask]
            mask_for_loss = data['mask_chain'][center_mask]
            if mask_for_loss.sum() == 0:
                return torch.tensor(0.0).to(logits.device)
            # print('mask_for_loss:',mask_for_loss.sum().item(),mask_for_loss.shape[0])
            S_onehot = F.one_hot(target, num_classes=logits.size(-1)).float()
            S_onehot = S_onehot + 0.1 / float(S_onehot.size(-1))
            S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(S_onehot * log_probs).sum(-1)
            loss_av = torch.sum(loss*mask_for_loss) / torch.sum(mask_for_loss)
            return loss_av
        else:
            center_mask = data["is_center"] & data["is_protein"]
            if "backbone_mask" in data:
                center_mask = center_mask & data["backbone_mask"]
            target = data["residue_token"][center_mask]
            loss = F.cross_entropy(logits, target)
            return loss

    def sc_packing(self,samples,pdb_path,t,sample_save_folder,models,infer_cfg,device):    
        protein_name = pdb_path.split('/')[-1].replace('.pdb','')+'_0'
        decode_mapping = {j:i for i,j in residue_tokens.items()}
        restype_3to1 = {v:k for k,v in restype_1to3.items()}
        samples[0][samples[0] > 21] = 10 #replace the non standard amino acid with GlY
        seq = ''.join([restype_3to1[decode_mapping[i.item()]] for i in samples[0]])

        proteins = replace_protein_sequence(vars(from_pdb_file(pdb_path, mse_to_met=True,ignore_non_std = False)), pdb_path, [[seq]])
        proteins = [(protein[0], transform_structure(protein[1], sc_d_mask_from_seq=True)) for protein in proteins]

        batch = collate_fn([proteins[0][1]])

        sample_results = sample_epoch(models, batch, infer_cfg['temperature'], device, n_recycle=1, resample=False, resample_args=infer_cfg['resample_args'])


        protein_strings = pdbs_from_prediction(sample_results)
        save_path = os.path.join(sample_save_folder, f'side_chain_t={round(t,3)}')
        if os.path.exists(save_path) == False:
            os.makedirs(save_path,exist_ok=True)

        
        with open(os.path.join(save_path, protein_name + '.pdb'), 'w') as f:
            f.write(protein_strings[0])

        make_merged_pdb(pdb_path, os.path.join(save_path, protein_name + '.pdb'))
        data = pdb2data(os.path.join(save_path, protein_name + '.pdb'),next(self.model.parameters()).device) 

        return data



    def test(self, dataloader):
        device = next(self.model.parameters()).device
        error = 0
        total_flatten_logits = torch.tensor([])
        total_flatten_true = torch.tensor([])
        total_flatten_interact_non_protein_logits = torch.tensor([])
        total_flatten_interact_non_protein_true = torch.tensor([])

        self.model.eval()
        for data in dataloader:
            try:
                data = {k: v.to(device) for k, v in data.__dict__.items()}
                data = {
                    k: v.float() if v.dtype == torch.float64 else v
                    for k, v in data.items()
                }
                noisy_data = {k: v.clone() for k, v in data.items()}
                noisy_data["residue_token"] = data["noisy_residue_token"]
                times = data["time_step"]
                flatten_logits, _ = self.model(noisy_data, times)

                center_mask = data["is_center"] & data["is_protein"]
                if "backbone_mask" in data:
                    center_mask = center_mask & data["backbone_mask"]
                flatten_true = data["residue_token"][center_mask]

                interact_non_protein = data["interact_non_protein_res"][center_mask].cpu()
                mask_for_loss = data['mask_chain'][center_mask].cpu()
                if mask_for_loss.sum() == 0:
                    continue
                flatten_logits = flatten_logits.detach().cpu()
                flatten_true = flatten_true.detach().cpu()
                total_flatten_logits = torch.cat((total_flatten_logits, flatten_logits[mask_for_loss]))
                total_flatten_true = torch.cat((total_flatten_true, flatten_true[mask_for_loss]))

                total_flatten_interact_non_protein_logits = torch.cat((total_flatten_interact_non_protein_logits, flatten_logits[interact_non_protein&mask_for_loss]))
                total_flatten_interact_non_protein_true = torch.cat((total_flatten_interact_non_protein_true, flatten_true[interact_non_protein&mask_for_loss]))
                del flatten_logits, flatten_true

            except Exception as e:
                error +=1
                print(f"Error in test in {e}")
                
        loss = F.cross_entropy(total_flatten_logits, total_flatten_true.long())
        accuracy = (
            (total_flatten_logits.argmax(dim=-1) == total_flatten_true).float().mean()
        )
        perplexity = torch.exp(loss)

        interact_non_protein_loss = F.cross_entropy(total_flatten_interact_non_protein_logits, total_flatten_interact_non_protein_true.long())
        interact_non_protein_accuracy = (
            (total_flatten_interact_non_protein_logits.argmax(dim=-1) == total_flatten_interact_non_protein_true).float().mean()
        )
        interact_non_protein_perplexity = torch.exp(interact_non_protein_loss)


        del total_flatten_logits, total_flatten_true,total_flatten_interact_non_protein_logits, total_flatten_interact_non_protein_true
        return loss, accuracy, perplexity,interact_non_protein_accuracy, interact_non_protein_perplexity


    _CORRUPT_SKIP_FIELDS = frozenset({
        "noisy_residue_token", "interact_non_protein_res", "interact_ion_res",
        "interact_nucleotide_res", "interact_molecule_res", "is_mask",
    })

    def corrupt_data_by_sample(self, data, time, sample):
        """Corrupt data for one sampling step: inject current samples, remove sidechains of masked residues."""
        device = data["residue_token"].device
        noisy_data = {k: v.squeeze().clone() for k, v in data.items()}

        # Step 1: replace designable positions with current sample tokens
        designable_mask = noisy_data["is_center"].bool() & noisy_data["is_protein"].bool()
        if "backbone_mask" in noisy_data:
            designable_mask = designable_mask & noisy_data["backbone_mask"].bool()
        noisy_data["residue_token"][designable_mask] = sample

        # Step 2: which center residues are still <MASK>?
        center_tokens = noisy_data["residue_token"][noisy_data["is_center"].bool()]
        residue_is_masked = (center_tokens == residue_tokens["<MASK>"])

        # Step 3: propagate <MASK> from center to all atoms of masked residues
        is_protein = noisy_data["is_protein"].bool()
        res_idx_protein = noisy_data["residue_index"][is_protein]
        if res_idx_protein.numel() > 0:
            assert center_tokens.shape[0] > res_idx_protein.max().item()
        atom_masked = residue_is_masked[res_idx_protein]
        noisy_data["residue_token"][is_protein] = torch.where(
            atom_masked,
            torch.tensor(residue_tokens["<MASK>"], device=device),
            noisy_data["residue_token"][is_protein],
        )

        # Step 4: build keep_mask — remove sidechain atoms of masked residues
        #   backbone atoms are always kept; sidechain atoms only kept if residue is unmasked
        keep_mask = torch.ones_like(noisy_data["residue_token"], dtype=torch.bool)
        sidechain_protein = ~noisy_data["is_backbone"].bool() & is_protein
        keep_mask[sidechain_protein] = ~residue_is_masked[noisy_data["residue_index"][sidechain_protein]]

        # Step 5: apply keep_mask to all per-atom fields
        time_tensor = torch.ones((1, 1), device=device) * time
        for name, item in noisy_data.items():
            if name == "time_step":
                noisy_data[name] = time_tensor
            elif name not in self._CORRUPT_SKIP_FIELDS:
                noisy_data[name] = item[keep_mask].unsqueeze(0)

        return data, noisy_data


    def _prepare_sampling_io(self, pdb_path):
        """Parse PDB/CIF, convert CIF->PDB if needed, set up save folder."""
        device = next(self.model.parameters()).device
        data = pdb2data(pdb_path, device)
        basename = os.path.basename(pdb_path)
        if 'cif' in pdb_path:
            stem = basename.replace('.cif.gz', '').replace('.cif', '')
            save_folder = os.path.join(self.sample_save_path, stem)
            os.makedirs(save_folder, exist_ok=True)
            cif_path = pdb_path
            pdb_path = os.path.join(save_folder, stem + '.pdb')
            cif_to_pdb(cif_path, pdb_path)
        else:
            stem = basename.replace('.pdb', '')
            save_folder = os.path.join(self.sample_save_path, stem)
            os.makedirs(save_folder, exist_ok=True)
            os.system(f'cp {pdb_path} {save_folder}')
        return data, pdb_path, save_folder, device

    def adaptive_sample(self, pdb_path, num_step=10, argmax_final=True,temp = 0.1,noise=1,threshold=0.8,regular_residue =True):

        self.model.eval()
        data, pdb_path, sample_save_folder, device = self._prepare_sampling_io(pdb_path)

        designable_mask = data["is_center"].bool() & data["is_protein"].bool()
        if "backbone_mask" in data:
            designable_mask = designable_mask & data["backbone_mask"].bool()
        true_tokens = data["residue_token"][designable_mask]
        if true_tokens.numel() == 0:
            raise ValueError("No designable protein residues (all have incomplete backbone or no protein)")
        t = 0.0
        sample_times = 0
        samples = (
            torch.ones_like(
                data["residue_token"][designable_mask],
                device=device,
            ).unsqueeze(0)
            * residue_tokens["<MASK>"]
        )
        B, T = samples.size()

        while t <= 1.0:
            data, noisy_data = self.corrupt_data_by_sample(data, t, samples)
            logits, _ = self.model(noisy_data, torch.tensor([[t]], device=device))

            rr =(logits.argmax(dim=1) == true_tokens).sum()/true_tokens.shape[0]
            print('time:', round(t,3), round(rr.item(),4),'context_num',noisy_data['residue_token'].shape[1])

            if round(t,3) >= 1.0 or sample_times >= num_step:
                if argmax_final:
                    samples_final = logits.argmax(dim=-1).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]
                else:
                    samples_final = torch.multinomial(
                        F.softmax(logits, dim=-1).view(B * T, -1), num_samples=1
                    ).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]

                print('final rr:',(samples == true_tokens).sum().item()/true_tokens.shape[0])
                return samples,logits

            pt_x1_probs = F.softmax(logits/temp, dim=-1)  # (B, T, V_size)

            conserve_mask = (pt_x1_probs.max(dim=-1)[0] > threshold).view(B, T)
            conserve_sample = pt_x1_probs.argmax(dim=-1).view(B, T)
            next_t = (conserve_mask.sum()/conserve_mask.shape[1]).item()
            dt = next_t - t

            # --- Forward transition: masked positions get probability of each token ---
            sample_is_mask = (samples == residue_tokens["<MASK>"]).view(B, T, 1).float()
            step_probs = dt * pt_x1_probs * ((1 + noise * t) / (1 - t))
            step_probs = step_probs * sample_is_mask

            # --- Backward noise: unmasked positions get probability of re-masking ---
            mask_onehot = F.one_hot(
                torch.tensor(residue_tokens["<MASK>"]),
                num_classes=logits.size(-1),
            ).view(1, 1, -1).to(device)
            step_probs += dt * (1 - sample_is_mask) * mask_onehot * noise

            # --- Normalize: remaining mass assigned to current token ---
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = 0.0
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = (
                1.0 - torch.sum(step_probs, dim=-1).flatten()
            )
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

            no_mask_prob = step_probs.clone()
            no_mask_prob[0,:,residue_tokens['<MASK>']] = 0  
            if (no_mask_prob.sum(dim=-1) <= 0).sum() > 0:  
                
                no_mask_prob[no_mask_prob.sum(dim=-1) != 1] = pt_x1_probs.view(B, T,-1)[no_mask_prob.sum(dim=-1) != 1].to(no_mask_prob.dtype)


            samples = torch.multinomial(
                no_mask_prob.view(-1, logits.size(-1)), num_samples=1
            ).view(B, T)
            samples = samples * ~conserve_mask + conserve_sample * conserve_mask

            if hasattr(self, 'sc'):
                data  = self.sc_packing(samples,pdb_path,t,sample_save_folder,models=self.sc,infer_cfg=self.infer_cfg,device=device)

            samples[~conserve_mask] = residue_tokens['<MASK>']
            t = t + dt
            sample_times += 1




    def sample(self, pdb_path, dt=0.1, argmax_final=True,noise=1,mask_interact=False):
        self.model.eval()
        data, pdb_path, sample_save_folder, device = self._prepare_sampling_io(pdb_path)
        designable_mask = data["is_center"].bool() & data["is_protein"].bool()
        if "backbone_mask" in data:
            designable_mask = designable_mask & data["backbone_mask"].bool()
        true_tokens = data["residue_token"][designable_mask]
        if true_tokens.numel() == 0:
            raise ValueError("No designable protein residues (all have incomplete backbone or no protein)")
        t = 0.0
        samples = (
            torch.ones_like(
                data["residue_token"][designable_mask],
                device=device,
            ).unsqueeze(0)
            * residue_tokens["<MASK>"]
        )
        B, T = samples.size()
        interact_res_index = data['residue_index'][data['is_protein']][data['interact_non_protein_res']].unique()

        while t <= 1.0:
            data, noisy_data = self.corrupt_data_by_sample(data, t, samples)
            logits, _ = self.model(noisy_data, torch.tensor([[t]], device=device))

            rr = (logits.argmax(dim=1) == true_tokens).sum()/true_tokens.shape[0]
            print('time:', round(t,3), round(rr.item(),4),'context_num',noisy_data['residue_token'].shape[1])

            if round(t,3) >= 1.0 or dt >=1.0:
                if argmax_final:
                    samples_final = logits.argmax(dim=-1).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]
                else:
                    samples_final = torch.multinomial(
                        F.softmax(logits, dim=-1).view(B * T, -1), num_samples=1
                    ).view(B, T)
                    samples_mask = samples == residue_tokens["<MASK>"]
                    samples[samples_mask] = samples_final[samples_mask]

                print('final rr:',(samples == true_tokens).sum().item()/true_tokens.shape[0])
                return samples,logits

            pt_x1_probs = F.softmax(logits, dim=-1)  # (B, T, V_size)

            # --- Forward transition: masked positions get probability of each token ---
            sample_is_mask = (samples == residue_tokens["<MASK>"]).view(B, T, 1).float()
            step_probs = (
                dt * pt_x1_probs * ((1 + noise * t) / ((1 - t)))
            )  # (B, T, V_size)

            step_probs = step_probs * sample_is_mask

            # --- Backward noise: unmasked positions get probability of re-masking ---
            mask_onehot = F.one_hot(
                torch.tensor(residue_tokens["<MASK>"]),
                num_classes=logits.size(-1),
            ).view(1, 1, -1).to(device)
            step_probs += dt * (1 - sample_is_mask) * mask_onehot * noise

            # --- Normalize: remaining mass assigned to current token ---
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = 0.0
            step_probs[
                torch.arange(B, device=device).repeat_interleave(T),
                torch.arange(T, device=device).repeat(B),
                samples.flatten(),
            ] = (
                1.0 - torch.sum(step_probs, dim=-1).flatten()
            )
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

            no_mask_prob = step_probs.clone()
            no_mask_prob[0,:,residue_tokens['<MASK>']] = 0

            if argmax_final:
                samples = no_mask_prob.argmax(dim=-1).view(B, T)
            else:
                samples = torch.multinomial(
                    no_mask_prob.view(-1, logits.size(-1)), num_samples=1
                ).view(B, T)

            if hasattr(self, 'sc'):
                data  = self.sc_packing(samples,pdb_path,t,sample_save_folder,models=self.sc,infer_cfg=self.infer_cfg,device=device)

            sample_mask = torch.multinomial(
                step_probs.view(-1, logits.size(-1)), num_samples=1
            ).view(B, T) == residue_tokens['<MASK>']

            samples[sample_mask] = residue_tokens['<MASK>']

            if mask_interact:
                samples[:,interact_res_index] = residue_tokens['<MASK>']

            t = t + dt


    def forward(self, data):
        """
        idx is the corrupted tokens (b, t)
        time is the time in the corruption process (b,)
        targets is the clean data (b, t)
        target_mask is 1.0 for points in the sequence that have been corrupted
            and should have loss calculated on them (b, t)
        do_self_cond_loop is whether to do two passes to train the self conditioning
        """

        noisy_data = {k: v.clone() for k, v in data.items()}
        noisy_data["residue_token"] = data["noisy_residue_token"]
        times = data["time_step"]

        b, t = noisy_data["residue_token"].size()
        assert (times < 1.1).all()  # 0 to 1 not 0 to 1000

        logits, _ = self.model(noisy_data, times)
        loss = self.compute_loss(logits, data)
        return logits, loss
