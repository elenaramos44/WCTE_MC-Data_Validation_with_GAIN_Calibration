import os
import glob
import json

folder = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2306"
files = glob.glob(os.path.join(folder, "card*_slot*_ch*_pos*_part*_chunk*.npz"))

pmts = set()

for f in files:
    base = os.path.basename(f)
    parts = base.split("_")

    card_id = int(parts[0][4:])   # cardXX
    slot_id = int(parts[1][4:])   # slotYY
    ch_id   = int(parts[2][2:])   # chZZ
    pos_id  = int(parts[3][3:])   # posWW

    pmts.add((card_id, slot_id, ch_id, pos_id))

pmts = sorted(list(pmts))

outpath = os.path.join(folder, "pmts_list.json")
with open(outpath, "w") as f:
    json.dump(pmts, f)

print(f"Saved {len(pmts)} PMTs to {outpath}")
