import csv
import re

input_csv = "/dapustor/nilufer/ADFLIP/pMT/pMT engineering consolidation_v4.xlsx - pMT sequences (1).csv"
output_csv = "/dapustor/nilufer/ADFLIP/pMT/pMT_stage_sequence_activity.csv"

stage_pattern = re.compile(r"stage\s*(\d+)", re.IGNORECASE)

rows_out = []
current_stage = None

with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        if len(row) < 4:
            continue

        first_col = row[0].strip()
        sequence = row[2].strip()
        activity = row[3].strip()

        match = stage_pattern.search(first_col)
        if match:
            stage_num = int(match.group(1))
            if 1 <= stage_num <= 10:
                current_stage = f"Stage {stage_num}"
            else:
                current_stage = None

        if current_stage and sequence:
            new_sequence = sequence[9:] + "L" + sequence[8:] + "L"
            rows_out.append([current_stage, new_sequence, activity])

with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Stage", "Sequence", "Activity"])
    writer.writerows(rows_out)

print(f"Saved {len(rows_out)} rows to {output_csv}")
