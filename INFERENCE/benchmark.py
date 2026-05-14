

import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from data import all_atom_parse as aap
from data.all_atom_parse import index_to_token,restype_3to1
from data.residue_config import configure as configure_residues
import argparse
from model.discrete_flow_aa import DiscreteFlow_AA
from model.zoidberg.zoidberg_GNN import Zoidberg_GNN




from ema_pytorch import EMA

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


def resolve_benchmark_data_path(pdb_type: str) -> str:
    candidates = {
         "test_MPNN": [
            "/dapustor/nilufer/ADFLIP/dataset/test_MPNN",
        ],
        
        "train_subset": [
            "/dapustor/nilufer/ADFLIP/dataset/train_subset",
        ],
        "test_set_109": [
            "/dapustor/nilufer/ADFLIP/dataset/test_set_109",
        ],
        "metal": [
            "/dapustor/nilufer/ADFLIP/dataset/test_metal_assembly1",
            "/dapustor/nilufer/ADFLIP/dataset/test_metal",
        ],
        "nucleotide": [
            "/dapustor/nilufer/ADFLIP/dataset/test_nucleotide_assembly1",
            "/dapustor/nilufer/ADFLIP/dataset/test_nucleotide",
        ],
        "molecule": [
            "/dapustor/nilufer/ADFLIP/dataset/test_small_molecule_assembly1",
            "/dapustor/nilufer/ADFLIP/dataset/test_small_molecule",
        ],
    }
    if pdb_type not in candidates:
        raise ValueError(f"Unsupported pdb_type: {pdb_type}")
    for path in candidates[pdb_type]:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"No benchmark dataset directory found for {pdb_type}. Tried: {candidates[pdb_type]}"
    )


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


def interact_residue(data):
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

    interact_non_protein_res = propagate_mask_vectorized(interact_non_protein,data.residue_index[data.is_protein])
    interact_ion_res = propagate_mask_vectorized(interact_ion,data.residue_index[data.is_protein])
    interact_nucleotide_res = propagate_mask_vectorized(interact_nucleotide,data.residue_index[data.is_protein])
    interact_molecule_res = propagate_mask_vectorized(interact_molecule,data.residue_index[data.is_protein])


    data.interact_non_protein_res = interact_non_protein_res.numpy()
    data.interact_ion_res = interact_ion_res.numpy()
    data.interact_nucleotide_res = interact_nucleotide_res.numpy()
    data.interact_molecule_res = interact_molecule_res.numpy()

    return data

def pdb2data(pdb_file,device = 'cpu',ligand_center=False,ion_center=False):
    data = aap.parse_mmcif_to_structure_data(pdb_file,ligand_center=ligand_center,ion_center=ion_center)
    data = interact_residue(data)
    result = {}
    for k, v in data.__dict__.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, (int, float)):
            result[k] = torch.Tensor([v])
        # skip non-tensor fields (assembly info lists/dicts)
    result['batch_index'] = torch.zeros_like(result['residue_index'])
    result = {
        k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v for k, v in result.items()
    }
    return result



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ADFLIP_v1.pt",
        help="pretrained model path",
    )
    parser.add_argument(
        "--pdb_type",
        type=str,
        default="metal",
        help='which pdb complex to test,protein,ion,nucleotide,molecule'
    )
    parser.add_argument(
        "--sample_scheme",
        type=str,
        default="adaptive",
        help='sampling scheme,fixed or adaptive'
    )
    parser.add_argument(
        '--noise_interact',
        type=float,
        default=1,
        help='noise add on mask token (dt*noise) on interaction position, '
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=1.0,
        help='time step for fixed sampling'
    )
    parser.add_argument(
        '--temp',
        type=float,
        default=0.1,
        help='temperature for adaptive sampling'
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.1,
        help='noise for adaptive sampling'
    )
    parser.add_argument(
        '--num_step',
        type=int,
        default=8,
        help='number of step for adaptive sampling'
    )
    parser.add_argument(
        '--thershold',
        type=float,
        default=0.5,
        help='thershold for adaptive sampling'
    )
    parser.add_argument(
        '--argmax_final',
        type=int,
        choices=[0, 1],
        default=0,
        help='whether to use argmax at final step (1=True, 0=False)'
    )
    parser.add_argument(
        '--save_tag',
        type=str,
        default='',
        help='optional extra suffix for output filename'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help='number of sampling repeats per pdb'
    )
    parser.add_argument(
        '--pdbs',
        type=str,
        nargs='+',
        default=None,
        help='only test specific PDB files, e.g. --pdbs 5l70.cif.gz 2vxx.cif.gz'
    )


    args = parser.parse_args()    

    ckpt_path = os.path.join('results','weights',args.ckpt_path)
    sample_scheme = args.sample_scheme
    pdb_type = args.pdb_type
    dt = args.dt
    temp = args.temp
    noise = args.noise
    num_step = args.num_step
    thershold = args.thershold
    argmax_final = bool(args.argmax_final)
    save_tag = args.save_tag.strip()
    n_samples = args.n_samples
    if n_samples < 1:
        raise ValueError('--n_samples must be >= 1')

    data_path = resolve_benchmark_data_path(pdb_type)

    if sample_scheme == 'fixed':
        save_name = pdb_type+'_'+sample_scheme+'_dt_'+f'{dt}'+'_temp_'+f'{temp}'+f'_noise_{args.noise}'+f'_argmax_{int(argmax_final)}'+f'_ns_{n_samples}'
    else:
        save_name = pdb_type+'_'+sample_scheme+'_step_'+f'{num_step}'+'_temp_'+f'{temp}'+f'_noise_{args.noise}'+f'_ther_{args.thershold}'+f'_argmax_{int(argmax_final)}'+f'_ns_{n_samples}'+'_tfmr'
    if save_tag:
        save_name = save_name + '_' + save_tag
    
    if not os.path.exists('/dapustor/nilufer/ADFLIP/results/benchmark/'+ckpt_path.split('/')[-1].replace('.pt','')+'/'):
        os.makedirs('/dapustor/nilufer/ADFLIP/results/benchmark/'+ckpt_path.split('/')[-1].replace('.pt','')+'/')
    
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    config.training.label_smoothing = True
    configure_residues(include_nonstd_amino_acids=getattr(config.data, 'include_nonstd_amino_acids', True))

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
        update_atom = config.zoidberg_denoiser.update_atom,
        num_decoder_blocks=config.zoidberg_denoiser.num_decoder_blocks,
        num_tfmr_heads=config.zoidberg_denoiser.num_tfmr_heads,
        num_tfmr_layers=config.zoidberg_denoiser.num_tfmr_layers,
        number_ligand_atom = config.zoidberg_denoiser.number_ligand_atom,
        mpnn_cutoff=config.zoidberg_denoiser.mpnn_cutoff,
        output_dim=config.zoidberg_denoiser.output_dim,
    )


    flow = DiscreteFlow_AA(config, denoiser, min_t=0.0,sidechain_packing=True,sample_save_path = 'results/finetune_benchmark/')
    flow.load_state_dict(ckpt["model"])

    ema_flow = EMA(
        flow,
        beta=config.training.ema_beta,
        update_every=config.training.ema_update_every,
    )
    ema_flow.load_state_dict(ckpt["ema"])
    ema_flow = ema_flow.to(device)


    if args.pdbs:
        test_pdb_list = args.pdbs
    else:
        test_pdb_list = os.listdir(data_path)

    record = {}
    total_len = []  

    for pdb_code in tqdm(test_pdb_list):
        if pdb_code.endswith('.png'):
            continue
        pdb = pdb_code
        # pdb = '1f35.pdb'
        print('pdb:',pdb)
        rr_list = [] 
        S_pred_list = []   
        interaction_nonprotein_list = []
        interact_ion_list = []
        interact_nucleotide_list = []
        interaction_molecule_list = []
        perplexity_list = []
        logits_list = []
        try:#
            data = pdb2data(os.path.join(data_path,pdb),device,ligand_center=False,ion_center=False)

            true_tokens = data["residue_token"][data["is_center"] & data["is_protein"]]
            interact_non_protein = data["interact_non_protein_res"][data["is_center"][data["is_protein"]]].cpu()
            interact_ion = data["interact_ion_res"][data["is_center"][data["is_protein"]]].cpu()
            interact_nucleotide = data["interact_nucleotide_res"][data["is_center"][data["is_protein"]]].cpu()
            interact_molecule = data["interact_molecule_res"][data["is_center"][data["is_protein"]]].cpu()

            for i in range(n_samples):
                if sample_scheme == 'fixed':
                    samples,logits = ema_flow.ema_model.sample(
                        os.path.join(data_path,pdb),
                        dt=dt,
                        argmax_final=argmax_final,
                        noise=noise,
                        mask_interact=False,
                    )
                elif sample_scheme == 'adaptive':
                    samples,logits = ema_flow.ema_model.adaptive_sample(os.path.join(data_path,pdb),num_step=num_step,argmax_final=argmax_final,temp=temp,noise = noise,threshold=thershold)
                samples = samples.squeeze()
                rr = ( (samples == true_tokens).sum()/true_tokens.shape[0]).item()
                rr_list.append(rr)
                interact_non_protein_rr = ( (samples[interact_non_protein] == true_tokens[interact_non_protein]).sum()/interact_non_protein.sum()).item()

                print(pdb,rr,interact_non_protein_rr)

                interaction_nonprotein_list.append(interact_non_protein_rr)
                interact_ion_rr = ( (samples[interact_ion] == true_tokens[interact_ion]).sum()/interact_ion.sum()).item()
                interact_ion_list.append(interact_ion_rr)
                interact_nucleotide_rr = ( (samples[interact_nucleotide] == true_tokens[interact_nucleotide]).sum()/interact_nucleotide.sum()).item()
                interact_nucleotide_list.append(interact_nucleotide_rr)
                interact_molecule_rr = ( (samples[interact_molecule] == true_tokens[interact_molecule]).sum()/interact_molecule.sum()).item()
                interaction_molecule_list.append(interact_molecule_rr)

                seq = ''.join([restype_3to1[index_to_token[res]] if 1 < res < 22 else 'X' for res in samples.squeeze().cpu().numpy()])
                S_pred_list.append(seq)
                seq_true = ''.join([restype_3to1[index_to_token[res]] if 1 < res < 22 else 'X' for res in true_tokens.squeeze().cpu().numpy()])
                logits_list.append(logits)

            ensemble_logits = torch.stack(logits_list).mean(0)
            perplexity = F.cross_entropy(ensemble_logits,true_tokens)
            record[pdb.replace('.pdb','')] = {'rr':rr_list,'S_pred':S_pred_list,'S_true':seq_true,'non_protein_rr':interaction_nonprotein_list,'ion_rr':interact_ion_list,'nucleotide_rr':interact_nucleotide_list,'molecule_rr':interaction_molecule_list,'perplexity':perplexity,'ensemble_logits':ensemble_logits,'non_protein_mask':interact_non_protein,'ion_mask':interact_ion,'nucleotide_mask':interact_nucleotide,'molecule_mask':interact_molecule}
        except Exception as e:
            print('error:',pdb,e)

            record[pdb.replace('.pdb','')] = e

        
        torch.save(record,'/dapustor/nilufer/ADFLIP/results/benchmark/'+ckpt_path.split('/')[-1].replace('.pt','')+'/'+save_name+'.pt')


    # Collect error information
    error_pdbs = {}
    success_count = 0
    for pdb_name, result in record.items():
        if not isinstance(result, dict):
            error_pdbs[pdb_name] = str(result)
        else:
            success_count += 1

    # Print mean statistics for non-protein interactions
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total PDBs: {len(record)}")
    print(f"Success: {success_count}")
    print(f"Errors: {len(error_pdbs)}")
    print("="*60)

    if pdb_type == 'metal':
        ion_rr_values = []
        for pdb_name, result in record.items():
            if isinstance(result, dict) and 'ion_rr' in result:
                ion_rr_values.extend(result['ion_rr'])

        # Filter out NaN values
        ion_rr_values = [x for x in ion_rr_values if not np.isnan(x)]

        if ion_rr_values:
            mean_ion_rr = np.mean(ion_rr_values)
            median_ion_rr = np.median(ion_rr_values)
            print(f"Metal (Ion) Interaction Statistics:")
            print(f"  Mean Ion Recovery Rate: {mean_ion_rr:.4f}")
            print(f"  Median Ion Recovery Rate: {median_ion_rr:.4f}")
            print(f"  Valid samples (with interactions): {len(ion_rr_values)}")
        else:
            print("No valid ion interaction data found (all samples have NaN values).")

    elif pdb_type == 'nucleotide':
        nucleotide_rr_values = []
        for pdb_name, result in record.items():
            if isinstance(result, dict) and 'nucleotide_rr' in result:
                nucleotide_rr_values.extend(result['nucleotide_rr'])

        # Filter out NaN values
        total_samples = len(nucleotide_rr_values)
        nucleotide_rr_values = [x for x in nucleotide_rr_values if not np.isnan(x)]

        if nucleotide_rr_values:
            mean_nucleotide_rr = np.mean(nucleotide_rr_values)
            median_nucleotide_rr = np.median(nucleotide_rr_values)
            print(f"Nucleotide Interaction Statistics:")
            print(f"  Mean Nucleotide Recovery Rate: {mean_nucleotide_rr:.4f}")
            print(f"  Median Nucleotide Recovery Rate: {median_nucleotide_rr:.4f}")
            print(f"  Valid samples (with interactions): {len(nucleotide_rr_values)}")
            print(f"  Samples without interactions (NaN): {total_samples - len(nucleotide_rr_values)}")
        else:
            print(f"No valid nucleotide interaction data found.")
            print(f"All {total_samples} samples have no nucleotide-interacting residues (within 5Å).")

    elif pdb_type == 'molecule':
        molecule_rr_values = []
        for pdb_name, result in record.items():
            if isinstance(result, dict) and 'molecule_rr' in result:
                molecule_rr_values.extend(result['molecule_rr'])

        # Filter out NaN values
        molecule_rr_values = [x for x in molecule_rr_values if not np.isnan(x)]

        if molecule_rr_values:
            mean_molecule_rr = np.mean(molecule_rr_values)
            median_molecule_rr = np.median(molecule_rr_values)
            print(f"Small Molecule Interaction Statistics:")
            print(f"  Mean Molecule Recovery Rate: {mean_molecule_rr:.4f}")
            print(f"  Median Molecule Recovery Rate: {median_molecule_rr:.4f}")
            print(f"  Valid samples (with interactions): {len(molecule_rr_values)}")
        else:
            print("No valid molecule interaction data found (all samples have NaN values).")

    print("="*60)

    # Print error details
    if error_pdbs:
        print("\nERROR DETAILS")
        print("="*60)
        for pdb_name, error_msg in error_pdbs.items():
            print(f"  {pdb_name}: {error_msg}")
        print("="*60)






        # torch.save(record,'results/benchmark/'+save_name+'_'+pdb_type+'_'+sample_scheme+'_noise_'+f'{args.noise_interact}'+'.pt')





    
# 6qi7: 149 residues
