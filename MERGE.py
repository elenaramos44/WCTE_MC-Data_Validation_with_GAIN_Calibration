#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse
import glob

def merge_pmts(folder, pmts_json, start_idx, end_idx):
    # Load list of PMTs
    with open(pmts_json, "r") as f:
        pmts = json.load(f)  # list of (card_id, slot_id, ch_id)
    
    pmts_to_process = pmts[start_idx:end_idx+1]

    for card_id, slot_id, ch_id in pmts_to_process:
        all_waveforms = []

        # Find all part0 chunks for this PMT
        part0_files = sorted(
            glob.glob(os.path.join(folder, f"card{card_id}_slot{slot_id}_ch{ch_id}_part0_chunk*.npz")),
            key=lambda x: int(x.split("_chunk")[1].split(".npz")[0])
        )

        for f0 in part0_files:
            # Corresponding part1 file
            f1 = f0.replace("_part0_", "_part1_")
            w0 = np.load(f0, allow_pickle=True)["waveforms"]

            if os.path.exists(f1):
                w1 = np.load(f1, allow_pickle=True)["waveforms"]
            else:
                w1 = np.empty((0,) + w0.shape[1:])  # empty array if missing

            merged_chunk = np.concatenate([w0, w1], axis=0)
            all_waveforms.append(merged_chunk)

        if all_waveforms:
            merged_pmt = np.concatenate(all_waveforms, axis=0)
            outname = os.path.join(folder, f"card{card_id}_slot{slot_id}_ch{ch_id}_combined.npz")
            np.savez_compressed(outname, waveforms=merged_pmt)
            print(f"✔ PMT {card_id}_{slot_id}_{ch_id}: {merged_pmt.shape[0]} waveforms saved.")
        else:
            print(f"⚠️ PMT {card_id}_{slot_id}_{ch_id}: no waveforms found!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge part0 and part1 waveform chunks for PMTs")
    parser.add_argument("--folder", required=True, help="Folder containing .npz files")
    parser.add_argument("--pmt-json", required=True, help="JSON file with list of PMTs")
    parser.add_argument("--start", type=int, required=True, help="Start PMT index")
    parser.add_argument("--end", type=int, required=True, help="End PMT index")
    args = parser.parse_args()

    merge_pmts(args.folder, args.pmt_json, args.start, args.end)
