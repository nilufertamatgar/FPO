# import numpy as np
# from pathlib import Path

# npz_path = Path("/dapustor/nilufer/ADFLIP/dataset/pMT_parsed/pMT_ligand.npz")
# out_path = Path("/dapustor/nilufer/ADFLIP/meta/pMT_ligand_contents_auth.txt")

# data = np.load(npz_path, allow_pickle=True)

# lines = []
# lines.append(f"Loaded: {npz_path}")
# lines.append(f"Keys: {data.files}")
# lines.append("")

# for key in data.files:
#     value = data[key]
#     lines.append(f"Key: {key}")
#     lines.append(f"  type: {type(value)}")
#     lines.append(f"  dtype: {value.dtype}")
#     lines.append(f"  shape: {value.shape}")
#     if value.ndim == 0:
#         lines.append(f"  value: {value.item()}")
#     else:
#         lines.append(f"  preview: {value[:1000]}")
#     lines.append("")

# out_path.write_text("\n".join(lines), encoding="utf-8")
# print(f"Wrote summary to: {out_path}")

# # from data.all_atom_parse import token_to_index, index_to_token, restype_3to1

# # print(token_to_index["ALA"])
# # print(index_to_token[token_to_index["ALA"]])
# # print(restype_3to1["ALA"])

# # import numpy as np
# # from data.all_atom_parse import load_structure_data

# # npz_path = "/dapustor/nilufer/ADFLIP/dataset/valid_parsed/01/101d.npz"
# # structure = load_structure_data(npz_path)

# # print("has asym_id_to_chain_index:", hasattr(structure, "asym_id_to_chain_index"))
# # if hasattr(structure, "asym_id_to_chain_index"):
# #     print("asym_id_to_chain_index:", structure.asym_id_to_chain_index)

# # print("unique chain_id (all atoms):", np.unique(structure.chain_id))

# # center_mask = structure.is_center & structure.is_protein
# # if hasattr(structure, "backbone_mask"):
# #     center_mask = center_mask & structure.backbone_mask

# # print("unique chain_id (center protein):", np.unique(structure.chain_id[center_mask]))

# # for cid in np.unique(structure.chain_id[center_mask]):
# #     count = np.sum(structure.chain_id[center_mask] == cid)
# #     print(f"internal chain_id {cid}: center residues = {count}")

import numpy as np
import sys
from pathlib import Path

sys.path.append("/dapustor/nilufer/ADFLIP")
from data.all_atom_parse import index_to_token, restype_3to1

npz_path = Path("/dapustor/nilufer/ADFLIP/dataset/pMT_parsed/pMT_ligand.npz")
out_txt = Path("/dapustor/nilufer/ADFLIP/meta/pMT_ligand_sequence.txt")

data = np.load(npz_path, allow_pickle=True)

mask = data["is_center"].astype(bool) & data["is_protein"].astype(bool)
tokens = data["residue_token"][mask]

sequence = "".join(
    restype_3to1.get(index_to_token.get(int(tok), "<UNK>"), "X")
    for tok in tokens
)

out_txt.write_text(sequence + "\n", encoding="utf-8")

print(f"Wrote sequence to: {out_txt}")
print(sequence)
