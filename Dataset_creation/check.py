import csv
from pathlib import Path
from data.all_atom_parse import index_to_token

OUT = Path("/dapustor/nilufer/ADFLIP/Dataset_creation/logit_index_to_token.csv")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["logit_column", "token"])
    for i in range(len(index_to_token)):
        writer.writerow([i, index_to_token[i]])

print(f"Wrote: {OUT}")
