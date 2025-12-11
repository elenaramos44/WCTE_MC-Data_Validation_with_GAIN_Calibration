import os
import glob
import json

folder = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/waveforms_including_position"
files = glob.glob(os.path.join(folder, "card*_slot*_ch*_part0_chunk*.npz"))

pmts = set()
for f in files:
    base = os.path.basename(f)
    parts = base.split("_")
    card_id = int(parts[0][4:])
    slot_id = int(parts[1][4:])
    ch_id = int(parts[2][2:])
    pmts.add((card_id, slot_id, ch_id))

pmts = sorted(list(pmts))

with open(os.path.join(folder, "pmts_list.json"), "w") as f:
    json.dump(pmts, f)
