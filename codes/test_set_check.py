from pathlib import Path

root = Path("/dapustor/nilufer/ADFLIP/dataset/test_small_molecule_parsed")

num_npz = sum(1 for p in root.rglob("*.npz"))
print(f"Total .npz files: {num_npz}")
