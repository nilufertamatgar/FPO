import re

input_file = "/dapustor/nilufer/ADFLIP/pMT/pocket_indices.csv"
output_file = "/dapustor/nilufer/ADFLIP/pMT/pocket_residues_modified.csv"

with open(input_file, "r") as f:
    text = f.read()

def replace_residue(match):
    resname = match.group(1)
    num = int(match.group(2))

    if num > 53:
        new_num = num - 9
    else:
        new_num = num + 273

    return f"{resname}{new_num}"

# Finds patterns like ALA134, GLY14, etc. anywhere in the file
modified_text = re.sub(r"\b([A-Z]{3})(\d+)\b", replace_residue, text)

with open(output_file, "w") as f:
    f.write(modified_text)

print(f"Modified file written to: {output_file}")
