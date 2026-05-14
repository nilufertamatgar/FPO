import multiprocessing as mp
from data import all_atom_parse as aap
from data.all_atom_parse import dump_structure_data
from data.residue_config import configure as configure_residues
import glob
import os
from tqdm import tqdm
import argparse
from pathlib import Path

def process_file(raw_data):
    try:
        data = aap.parse_or_load_mmcif(raw_data,ion_center=False,ligand_center=False)
        input_path = Path(raw_data)
        pdb = input_path.name.split('.')[0]
        dataset_root = input_path.parent.parent
        split_name = input_path.parent.name
        out_root = dataset_root / f"{split_name}_parsed"
        subdir = pdb[1:3] if len(pdb) >= 3 else pdb[:2]
        final_path = str(out_root / subdir / f"{pdb}.npz")

        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        dump_structure_data(data, final_path)
    except Exception as e:
        print(f"Error processing {raw_data}: {str(e)}")

def main(data_path):
    raw_data_list = glob.glob(data_path + '*.cif.gz')

    # Determine the number of CPU cores to use
    num_cores = mp.cpu_count()

    # Create a pool of worker processes
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to create a progress bar
        for _ in tqdm(pool.imap_unordered(process_file, raw_data_list), total=len(raw_data_list)):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="train",
        help="Path to the data directory containing .cif.gz files"
    )
    parser.add_argument(
        "--include_nonstd_amino_acids",
        type=bool,
        default=True,
        help="Include non-standard amino acids in protein residue set"
    )
    args = parser.parse_args()
    configure_residues(include_nonstd_amino_acids=args.include_nonstd_amino_acids)
    data_path = args.data_path
    main(data_path)
