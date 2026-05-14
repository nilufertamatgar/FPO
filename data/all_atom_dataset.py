import torch
import glob
import os
import numpy as np
import pandas as pd
import torch.utils
import torch.utils.data
import random
import pickle
from tqdm import tqdm

from pathlib import Path
from typing import Dict, List, Literal

from data.all_atom_parse import residue_tokens
from data import all_atom_parse as aap
from data.all_atom_parse import StructureData
from data.utils import TimeSortedCacheRBT
from scipy.spatial import cKDTree


Partition = Literal["train", "valid", "test"]

def propagate_mask_vectorized(mask, index):
    # Ensure inputs are tensors
    mask = torch.as_tensor(mask, dtype=torch.bool)
    index = torch.as_tensor(index)

    # Get unique indices and their inverse mapping
    unique_indices, inverse_indices = torch.unique(index, return_inverse=True)

    # Create a tensor to hold the "any True" status for each unique index
    any_true = torch.zeros_like(unique_indices, dtype=torch.bool)

    # Use scatter_reduce to check if any mask is True for each unique index
    any_true.scatter_reduce_(0, inverse_indices, mask, reduce='amax')

    # Expand the any_true tensor to match the original shape
    return any_true[inverse_indices]

def _check_bounds(indices: np.ndarray, length: int, name: str):
    """Assert index bounds for array indexing."""
    if np.any(indices < 0) or np.any(indices >= length):
        raise IndexError(
            f"Index out of bounds for {name}: valid range [0, {length}), got indices min {indices.min()}, max {indices.max()}"
        )


def _apply_HIS_trim(seq_is_his: np.ndarray, res_ids: np.ndarray) -> np.ndarray:
    """Replicate poly-His trimming on a single chain."""
    L = int(seq_is_his.shape[0])
    if L == 0:
        return res_ids

    def _all_h(start: int, end: int) -> bool:
        if start < 0 or end > L:
            return False
        return bool(np.all(seq_is_his[start:end]))

    keep = res_ids
    # C-term then N-term
    if _all_h(L - 6, L):
        keep = keep[:-6]
    if _all_h(0, 6):
        keep = keep[6:]
    if _all_h(L - 7, L - 1):
        keep = keep[:-7]
    if _all_h(L - 8, L - 2):
        keep = keep[:-8]
    if _all_h(L - 9, L - 3):
        keep = keep[:-9]
    if _all_h(L - 10, L - 4):
        keep = keep[:-10]
    if _all_h(1, 7):
        keep = keep[7:]
    if _all_h(2, 8):
        keep = keep[8:]
    if _all_h(3, 9):
        keep = keep[9:]
    if _all_h(4, 10):
        keep = keep[10:]
    return keep


def _trim_poly_his_in_assembly(
    data: StructureData,
    assembly_chain_ids: List[int],
    drop_short_chain: bool = True,
) -> StructureData:
    """Trim poly-His tags on selected assembly chains by slicing StructureData."""
    his_idx = residue_tokens.get("HIS")
    if his_idx is None:
        return data
    if assembly_chain_ids is None:
        return data
    assembly_chain_ids = np.atleast_1d(assembly_chain_ids)
    if assembly_chain_ids.size == 0:
        return data

    center_mask = data.is_center & data.is_protein
    if hasattr(data, "backbone_mask"):
        center_mask = center_mask & data.backbone_mask
    center_mask = center_mask & np.isin(data.chain_id, assembly_chain_ids)
    if not np.any(center_mask):
        return data

    center_chain = data.chain_id[center_mask]
    center_resi = data.residue_index[center_mask]
    center_tok = data.residue_token[center_mask]
    is_his = center_tok == his_idx

    remove_residue_ids = set()
    for chain in np.unique(center_chain):
        sel = np.where(center_chain == chain)[0]
        if sel.size == 0:
            continue
        order = np.argsort(center_resi[sel])
        seq_is_his = is_his[sel][order]
        chain_res_ids = center_resi[sel][order]

        kept_res_ids = _apply_HIS_trim(seq_is_his, chain_res_ids.copy())
        if drop_short_chain and len(np.unique(kept_res_ids)) < 4:
            remove_residue_ids.update(np.unique(chain_res_ids).tolist())
        else:
            remove_residue_ids.update([rid for rid in chain_res_ids if rid not in set(kept_res_ids)])

    if not remove_residue_ids:
        return data

    keep_mask = ~np.isin(data.residue_index, list(remove_residue_ids))
    trimmed = aap.slice_structure_data(data, keep_mask)
    for k in ("asmb_ids", "asmb_chains", "asmb_chain_indices", "asym_id_to_chain_index"):
        if hasattr(data, k):
            setattr(trimmed, k, getattr(data, k))
    return trimmed




class AllAtomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        dataset_type: str = "train",
        use_cache: bool = True,
        cache_length: int = -1,
        test_time=0.0,
        max_num_residues=None,
        max_num_atoms=None,
    ):
        self.cfg = cfg
        if max_num_residues is None:
            self.max_num_residues = cfg.max_num_residues
        if max_num_atoms is None:
            self.max_num_atoms = cfg.max_num_atoms
        self.dataset_type   = dataset_type
        self.cut_position_type = cfg.cut_position_type
        self.random_gen = np.random.default_rng(cfg.random_seed)

        paths = cfg.data_path

        self.structs = (
            glob.glob(f"{paths}/*.pdb")
            + glob.glob(f"{paths}/*.cif")
            + glob.glob(f"{paths}/*.npz")
            + glob.glob(f"{paths}/*.cif.gz")
        )
        if len(self.structs) == 0:
            self.structs = (
                glob.glob(f"{paths}/**/*.pdb")
                + glob.glob(f"{paths}/**/*.cif")
                + glob.glob(f"{paths}/**/*.npz")
                + glob.glob(f"{paths}/**/*.cif.gz")
            )
        self.test_time = test_time
        self.use_cache = use_cache

        if self.use_cache:
            self.cache_data = TimeSortedCacheRBT(
                capacity=cache_length, parse_fn=aap.parse_or_load_mmcif
            )

    def __len__(self):
        return len(self.structs)

    def __getitem__(self, idx, mask_chain_id = None, data_override=None) -> aap.StructureData:
        try:
            if data_override is not None:
                data = data_override
            elif self.use_cache:
                data = self.cache_data.get(self.structs[idx])
            else:
                data = aap.parse_or_load_mmcif(self.structs[idx])
            if isinstance(data, tuple):
                data = data[1]
            # map chain labels (e.g., label_asym_id) to internal indices if needed
            if mask_chain_id is not None:
                mapped = []
                for m in np.atleast_1d(mask_chain_id):
                    if isinstance(m, (int, np.integer)):
                        mapped.append(int(m))
                        continue
                    m_str = str(m)
                    if m_str.isdigit():
                        mapped.append(int(m_str))
                        continue
                    if hasattr(data, "asym_id_to_chain_index"):
                        chain_map = getattr(data, "asym_id_to_chain_index")
                        if isinstance(chain_map, dict) and m_str in chain_map:
                            mapped.append(int(chain_map[m_str]))
                if len(mapped) == 0:
                    mapped = list(np.unique(data.chain_id))
                mask_chain_id = mapped
            num_residues = data.num_residues()


            if np.unique(data.residue_index)[-1]+1 != np.unique(data.residue_index).shape[0]:
                data.residue_index = np.unique(data.residue_index, return_inverse=True)[1]

            if num_residues > self.max_num_residues:
                # Here can do all kinds of complicated logic about finding ligands
                # that are closest to protein residues etc
                # For now, just pick a random protein residue
                center_chain_id = random.choice(mask_chain_id)
                chain_mask = data.chain_id == center_chain_id
                if self.cut_position_type == "protein":
                    random_position = self.random_gen.choice(
                        data.position[data.is_protein&chain_mask], axis=0
                    )
                elif self.cut_position_type == "interface":
                    # find atoms near the interface between different chains
                    unique_chains = np.unique(data.chain_id)
                    if len(unique_chains) > 1:
                        # for each atom, find nearest atom from a different chain
                        # interface atoms = those within 8A of another chain
                        interface_mask = np.zeros(len(data.position), dtype=bool)
                        for cid in unique_chains:
                            this_chain = data.chain_id == cid
                            other_chains = data.chain_id != cid
                            if not np.any(other_chains):
                                continue
                            tree = cKDTree(data.position[other_chains])
                            dists, _ = tree.query(data.position[this_chain])
                            interface_mask[this_chain] = dists < 8.0
                        if np.any(interface_mask):
                            random_position = self.random_gen.choice(
                                data.position[interface_mask], axis=0
                            )
                        else:
                            # fallback: no interface found, use random atom
                            random_position = self.random_gen.choice(
                                data.position, axis=0
                            )
                    else:
                        # single chain, just pick random atom
                        random_position = self.random_gen.choice(
                            data.position, axis=0
                        )
                elif self.cut_position_type == "ligand":
                    if data.is_protein.sum() == data.is_protein.shape[0]: # no ligand
                        random_position = self.random_gen.choice(
                            data.position, axis=0
                        )
                        # print("No ligand found, path", self.structs[idx])
                    else:
                        random_position = self.random_gen.choice(
                        data.position[~data.is_protein], axis=0
                    )

                data = aap.get_closest_n_residues(
                    data, random_position, n=self.max_num_residues
                )

            if len(data) > self.max_num_atoms:

                if data.is_protein.sum() == 0:
                    print("No protein found, path", self.structs[idx])
                    return None
                else:
                    random_position = self.random_gen.choice(
                        data.position[data.is_protein], axis=0
                    )
                    data = aap.get_closest_n_atoms(
                        data,
                        random_position,
                        n=self.max_num_atoms,
                    )

            if data.is_protein.sum() == 0:
                print("No protein found, path", self.structs[idx])
                return None
            data = self.interact_residue(data)

            if self.dataset_type == "train":
                return self.corrupt(data,mask_chain_id=mask_chain_id)
            else:
                return self.corrupt(data, time=np.array([self.test_time]),mask_chain_id=mask_chain_id)

        except Exception as e:
            print(f"Failed to parse {self.structs[idx]}", e)
            return None

    def corrupt(self, data: aap.StructureData, time=None, mask_chain_id=None) -> aap.StructureData:
        batch_dict = data.__dict__

        # Step 1: determine which chains to mask (designable chains)
        mask_chain = np.zeros_like(batch_dict['chain_id'], dtype=bool)
        if mask_chain_id is not None:
            mask_chain_id = np.atleast_1d(mask_chain_id)
            mask_chain[np.isin(batch_dict['chain_id'], mask_chain_id)] = True

        # Step 2: sample which center residues get corrupted
        num_residue_tokens = batch_dict['is_center'].sum()
        time_step = time if time is not None else np.random.uniform(0, 1, 1)
        u = np.random.uniform(0, 1, num_residue_tokens)
        residue_is_masked = u < (1.0 - time_step)
        # only mask residues in designable chains
        residue_is_masked[~mask_chain[batch_dict['is_center']]] = False

        # Step 3: propagate <MASK> from center to all atoms of masked residues
        noisy_residue_token = np.copy(batch_dict['residue_token'])
        protein_positions = np.nonzero(batch_dict['is_protein'])[0]
        _check_bounds(protein_positions, batch_dict['residue_index'].shape[0], 'residue_index (via protein_positions)')
        protein_center_idx = batch_dict['residue_index'][protein_positions]
        _check_bounds(protein_center_idx, residue_is_masked.shape[0], 'residue_is_masked (via protein_center_idx)')

        protein_atom_masked = residue_is_masked[protein_center_idx]
        atom_should_mask = np.zeros_like(batch_dict['is_protein'], dtype=bool)
        atom_should_mask[protein_positions] = protein_atom_masked
        noisy_residue_token[atom_should_mask] = residue_tokens['<MASK>']
        batch_dict['noisy_residue_token'] = noisy_residue_token

        # Step 4: build keep_mask — remove sidechain atoms of masked residues
        #   backbone atoms are always kept; sidechain atoms only kept if residue is unmasked
        keep_mask = np.ones_like(batch_dict['residue_token'], dtype=bool)
        sidechain_protein = (~batch_dict['is_backbone']) & batch_dict['is_protein']
        side_idx = batch_dict['residue_index'][sidechain_protein]
        _check_bounds(side_idx, residue_is_masked.shape[0], 'residue_is_masked (via sidechain indices)')
        keep_mask[sidechain_protein] = ~residue_is_masked[side_idx]

        # Step 5: apply keep_mask to all per-atom fields
        L = batch_dict['residue_token'].shape[0]
        if keep_mask.shape[0] != L:
            raise IndexError(f"Mask length mismatch: expected {L}, got {keep_mask.shape[0]}")

        for name, arr in list(batch_dict.items()):
            if name == 'noisy_residue_token':
                batch_dict[name] = noisy_residue_token[keep_mask]
            elif name == 'time_step':
                batch_dict[name] = time_step
            elif name == 'is_mask':
                batch_dict[name] = mask_chain[keep_mask]
            elif isinstance(arr, np.ndarray) and arr.shape[0] == L:
                batch_dict[name] = arr[keep_mask]

        # Step 6: reconstruct StructureData, preserving extra metadata (e.g., assemblies)
        update_mask_chain = mask_chain[keep_mask]
        core = {k: v for k, v in batch_dict.items() if k in aap.structure_data_fields}
        extra = {k: v for k, v in batch_dict.items() if k not in aap.structure_data_fields}
        data = aap.StructureData(**core)
        data.mask_chain = update_mask_chain
        for k, v in extra.items():
            setattr(data, k, v)

        # Debug diagnostics
        if getattr(self.cfg, "debug_corrupt", False):
            center_mask = batch_dict['is_center'] if isinstance(batch_dict['is_center'], np.ndarray) and batch_dict['is_center'].dtype == bool else batch_dict['is_center'].astype(bool)
            chain_center_mask = mask_chain[center_mask] if center_mask.shape[0] == mask_chain.shape[0] else np.zeros_like(residue_is_masked)
            masked_in_chain = residue_is_masked[chain_center_mask] if chain_center_mask.shape[0] == residue_is_masked.shape[0] else residue_is_masked
            frac = masked_in_chain.mean() if masked_in_chain.size > 0 else float("nan")
            print(
                f"[debug corrupt] t={float(time_step):.3f} centers={residue_is_masked.size} in_chain={chain_center_mask.sum()} "
                f"masked_in_chain={masked_in_chain.sum()} frac={frac:.3f}"
            )

        return data


    def interact_residue(self,data):
        protein_pos = np.expand_dims(data.position[data.is_protein],axis=1)
        non_protein_pos = np.expand_dims(data.position[~data.is_protein],axis=0)
        ion_pos = np.expand_dims(data.position[data.is_ion],axis=0)
        nucleotide_pos = np.expand_dims(data.position[data.is_nucleotide],axis=0)
        molecule_pos = np.expand_dims(data.position[~data.is_protein&~data.is_ion&~data.is_nucleotide],axis=0)

        dist_non_protein = np.sqrt((np.sum((protein_pos-non_protein_pos)**2,axis=-1)))
        interact_non_protein = np.any(dist_non_protein<5,axis=1)
        dist_ion = np.sqrt((np.sum((protein_pos-ion_pos)**2,axis=-1)))
        interact_ion = np.any(dist_ion<5,axis=1)
        dist_nucleotide = np.sqrt((np.sum((protein_pos-nucleotide_pos)**2,axis=-1)))
        interact_nucleotide = np.any(dist_nucleotide<5,axis=1)
        dist_molecule = np.sqrt((np.sum((protein_pos-molecule_pos)**2,axis=-1)))
        interact_molecule = np.any(dist_molecule<5,axis=1)


        interact_non_protein_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_non_protein_res[data.is_protein] = propagate_mask_vectorized(interact_non_protein,data.residue_index[data.is_protein])
        interact_ion_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_ion_res[data.is_protein] = propagate_mask_vectorized(interact_ion,data.residue_index[data.is_protein])
        interact_nucleotide_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_nucleotide_res[data.is_protein] = propagate_mask_vectorized(interact_nucleotide,data.residue_index[data.is_protein])
        interact_molecule_res = np.zeros(data.residue_index.shape[0],dtype=bool)
        interact_molecule_res[data.is_protein] = propagate_mask_vectorized(interact_molecule,data.residue_index[data.is_protein])


        data.interact_non_protein_res = interact_non_protein_res
        data.interact_ion_res = interact_ion_res
        data.interact_nucleotide_res = interact_nucleotide_res
        data.interact_molecule_res = interact_molecule_res

        return data


def _load_cluster_ids(path: Path | str) -> List[int]:
    """Return a list of integer IDs, one per line, from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f]

class Cluster_AllAtomDataset(AllAtomDataset):
    def __init__(
        self,
        cfg,
        dataset_type: str = "train",
        use_cache: bool = True,
        cache_length: int = -1,
        test_time=0.0,
        max_num_residues=None,
        max_num_atoms=None,
    ):
        super().__init__(cfg, dataset_type, use_cache, cache_length, test_time, max_num_residues, max_num_atoms)
        if max_num_residues is None:
            self.max_num_residues = cfg.max_num_residues
            self.max_num_atoms = cfg.max_num_atoms
        else:
            self.max_num_residues = max_num_residues
            self.max_num_atoms = max_num_atoms
        self.cfg = cfg
        assert dataset_type in ["train", "valid", "test"], f"Invalid dataset type: {dataset_type}"
        # prefer split-specific data paths if provided
        if dataset_type == "train":
            self.data_path = getattr(cfg, "train_data_path", cfg.data_path)
        elif dataset_type == "valid":
            self.data_path = getattr(cfg, "valid_data_path", cfg.data_path)
        else:
            self.data_path = getattr(cfg, "test_data_path", cfg.data_path)
        with open(cfg.cluster_path, "rb") as f:
            self.cluster_dict = pickle.load(f)

        # load cluster id lists; fall back to valid for test if not provided
        test_cluster_path = getattr(cfg, "test_cluster", None) or getattr(cfg, "validation_cluster", None)
        id_lists: Dict[Partition, List[int]] = {
            "train": _load_cluster_ids(cfg.train_cluster),
            "valid": _load_cluster_ids(cfg.validation_cluster),
            "test": _load_cluster_ids(test_cluster_path) if test_cluster_path else _load_cluster_ids(cfg.validation_cluster),
        }
        self.sel_keys = id_lists[dataset_type]
        self.select_cluster = {
            idx: value
            for idx, (key, value) in enumerate(self.cluster_dict.items())
            if key in self.sel_keys
        }

        # optionally filter out excluded PDBs (e.g. ProteinMPNN excluded set)
        excluded_pdbs_path = getattr(cfg, "excluded_pdbs_path", None)
        if excluded_pdbs_path:
            excluded_df = pd.read_csv(excluded_pdbs_path)
            excluded_set = set(excluded_df["PDB_IDS"].str.lower())
            n_before = sum(len(v) for v in self.select_cluster.values())
            filtered = {}
            for idx, chains in self.select_cluster.items():
                kept = [c for c in chains if c[:4].lower() not in excluded_set]
                if kept:
                    filtered[idx] = kept
            self.select_cluster = filtered
            self.sel_keys = list(self.select_cluster.keys())
            n_after = sum(len(v) for v in self.select_cluster.values())
            print(f"Excluded PDBs filter: {n_before} chains -> {n_after} chains "
                  f"({n_before - n_after} removed), "
                  f"{len(self.select_cluster)} clusters remaining")

        self.all_pdb = []
        for _, v in self.select_cluster.items():
            #v  17: ['5jog_0', '5joh_0', '5m5q_0', '4f7o_1', '4f7o_0'], 18: ['6jo8_2', '6ort_0', '6nk3_1', '6jo7_0', '6nk3_0']
            v_ = list(set([x[:4] for x in v]))
            self.all_pdb.extend(v_)
        if self.use_cache:
            self.structs = [
                os.path.join(
                    self.data_path,
                    x[1:3],
                    x + ".npz"
                ) for x in self.all_pdb
            ]
        else:
            self.structs = [
                os.path.join(
                    cfg.data_path,
                    x+ ".cif.gz"
                ) for x in self.all_pdb
            ]
        print(f"Dataset {dataset_type} contains {len(self.structs)} structures, {len(set(self.structs))} unique PDB IDs")
    def __len__(self) -> int:
        return len(self.select_cluster.keys())

    def __getitem__(self, idx) -> StructureData:
        cluster = self.select_cluster[self.sel_keys[idx]]
        sample = random.choice(cluster)
        # check if sample in this cluster has same pdb id
        sel_pdb = sample[:4]
        def _chain_label(s: str) -> str:
            parts = s.split("_", 1)
            return parts[1] if len(parts) > 1 else s[4:]
        sample_list = [x for x in cluster if x[:4] == sel_pdb]
        anchor_label = _chain_label(sample)
        if self.use_cache:
            sample_pdb_path = os.path.join(self.data_path, sample[1:3], sel_pdb + ".npz")
        else:
            sample_pdb_path = os.path.join(''.join(self.structs[0].split('/')[:-2]),sample[1:3],sel_pdb + ".cif.gz")
        index = self.structs.index(sample_pdb_path)

        # load data once to determine assembly + chain mapping
        if self.use_cache:
            data = self.cache_data.get(sample_pdb_path)
        else:
            data = aap.parse_or_load_mmcif(sample_pdb_path)
        if isinstance(data, tuple):
            data = data[1]

        chain_map = getattr(data, "asym_id_to_chain_index", {})
        if isinstance(chain_map, np.ndarray):
            if chain_map.dtype.kind == "O":
                if chain_map.size == 1:
                    chain_map = chain_map.item()
                elif chain_map.size > 0 and isinstance(chain_map.flat[0], dict):
                    chain_map = chain_map.flat[0]
        elif isinstance(chain_map, list) and len(chain_map) == 1 and isinstance(chain_map[0], dict):
            chain_map = chain_map[0]
        if not isinstance(chain_map, dict):
            chain_map = {}

        def _to_chain_index(label):
            if isinstance(label, (int, np.integer)):
                return int(label)
            s = str(label)
            if s.isdigit():
                return int(s)
            if isinstance(chain_map, dict) and s in chain_map:
                return int(chain_map[s])
            return None

        cluster_chain_labels = [_chain_label(x) for x in sample_list]
        cluster_chain_indices = [
            idx for idx in (_to_chain_index(lbl) for lbl in cluster_chain_labels) if idx is not None
        ]
        anchor_idx = _to_chain_index(anchor_label)

        # find assembly that contains anchor chain
        asmb_indices = None
        if anchor_idx is not None and hasattr(data, "asmb_chain_indices"):
            candidates = [
                idxs for idxs in getattr(data, "asmb_chain_indices", [])
                if anchor_idx in idxs
            ]
            if candidates:
                asmb_indices = random.choice(candidates)

        if asmb_indices is not None:
            mask_chain_id = [idx for idx in cluster_chain_indices if idx in asmb_indices]
        else:
            mask_chain_id = cluster_chain_indices

        if not mask_chain_id:
            mask_chain_id = [anchor_idx] if anchor_idx is not None else list(np.unique(data.chain_id))

        if getattr(self.cfg, "trim_his_tag", False):
            his_chain_ids = asmb_indices if asmb_indices is not None else mask_chain_id
            data = _trim_poly_his_in_assembly(
            data,
            his_chain_ids,
            drop_short_chain=getattr(self.cfg, "trim_his_tag_drop_short_chain", True),
        )

        if getattr(self.cfg, "debug_assembly_mask", False):
            asmb_ids = getattr(data, "asmb_ids", [])
            asmb_chain_indices = getattr(data, "asmb_chain_indices", [])
            anchor_asmb = None
            def _same_indices(a, b):
                if a is b:
                    return True
                try:
                    return np.array_equal(np.asarray(a), np.asarray(b))
                except Exception:
                    return False
            if asmb_indices is not None:
                for i, idxs in enumerate(asmb_chain_indices):
                    if _same_indices(idxs, asmb_indices):
                        anchor_asmb = asmb_ids[i] if i < len(asmb_ids) else i
                        break
            print("\n[debug assembly mask]")
            print("pdb:", sel_pdb, "sample:", sample, "anchor_label:", anchor_label, "anchor_idx:", anchor_idx)
            print("cluster_chain_labels:", cluster_chain_labels)
            print("cluster_chain_indices:", cluster_chain_indices)
            print("asmb_ids:", asmb_ids)
            print("asmb_chain_indices:", asmb_chain_indices)
            print("chosen_assembly:", anchor_asmb, "chosen_indices:", asmb_indices)
            print("final_mask_chain_id:", mask_chain_id)

        return super().__getitem__(index,mask_chain_id=mask_chain_id, data_override=data)


if __name__ == "__main__":

    # class Config:
    #     data_path = "/ceph/scheres_grp/kyi/Documents/dataset/ligand_pdb/train_parsed"
    #     max_num_atoms = 4096
    #     max_num_residues = 512
    #     random_seed = 42
    #     cut_position_type = "ligand"
    #     use_cathe = True

    # dataset = AllAtomDataset(Config())


    class Config:
        # parsed data paths (split-specific)
        train_data_path = "/ssd/dataset/pdb/train_parsed"
        valid_data_path = "/ssd/dataset/pdb/valid_parsed"
        test_data_path = "/ssd/dataset/pdb/test_parsed"

        # fallback single path (not used when split paths exist)
        data_path = "/ssd/dataset/pdb/valid_parsed"

        # new cluster files
        cluster_path = "/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/new_cluster/cluster_to_pdb_chain.pkl"
        train_cluster = "/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/new_cluster/train_clusters.txt"
        validation_cluster = "/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/new_cluster/valid_clusters.txt"
        test_cluster = "/ceph.groups/scheres_grp/kyi/Documents/ADFLIP/data/new_cluster/test_clusters.txt"

        max_num_atoms = 4096
        max_num_residues = 512
        random_seed = 42
        cut_position_type = "ligand"
        use_cathe = True
        debug_assembly_mask =  False
        debug_corrupt = True
        trim_his_tag = False
        trim_his_tag_drop_short_chain = True


    dataset = Cluster_AllAtomDataset(Config(), dataset_type="train", use_cache=True, cache_length=5000)
    print(f"valid dataset clusters: {len(dataset)}")

    num_checks = min(200, len(dataset))
    for i in tqdm(range(num_checks)):
        try:
            _ = dataset[i]
        except Exception as e:
            print(f"Failed to parse {dataset.structs[i]}", e)
