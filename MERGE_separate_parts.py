#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse
import glob
from collections import defaultdict

def merge_pmts_by_part(folder, pmts_json, start_idx, end_idx):
    # Load list of PMTs
    with open(pmts_json, "r") as f:
        pmts = json.load(f)  # list of (card_id, slot_id, ch_id, pos_id)
    
    # Safe slicing
    pmts_to_process = pmts[start_idx : min(end_idx+1, len(pmts))]

    if not pmts_to_process:
        print(f"No PMTs to process in range {start_idx}-{end_idx}.")
        return

    for card_id, slot_id, ch_id, pos_id in pmts_to_process:
        # Find all files for this PMT
        all_files = glob.glob(os.path.join(
            folder,
            f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}_part*_chunk*.npz"
        ))

        if not all_files:
            print(f"⚠️ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: no files found.")
            continue

        # Group files by part
        files_by_part = defaultdict(list)
        for f in all_files:
            base = os.path.basename(f)
            part_str = [s for s in base.split("_") if s.startswith("part")][0]
            part_id = int(part_str[4:])
            files_by_part[part_id].append(f)

        # Merge chunks **per part**
        for part_id, chunk_files in sorted(files_by_part.items()):
            chunk_files = sorted(chunk_files, key=lambda x: int(x.split("_chunk")[1].split(".npz")[0]))
            all_waveforms = []
            total_events = 0

            for fpath in chunk_files:
                try:
                    with np.load(fpath, allow_pickle=True) as data:
                        w = data.get("waveforms", None)
                        if w is not None and w.size > 0:
                            all_waveforms.append(w)
                            total_events += w.shape[0]
                        else:
                            print(f"  ⚠️ Empty waveforms in {os.path.basename(fpath)}")
                except Exception as e:
                    print(f"  ❌ Failed to load {os.path.basename(fpath)}: {e}")

            if all_waveforms:
                merged_pmt = np.concatenate(all_waveforms, axis=0)
                outname = os.path.join(
                    folder, f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}_part{part_id}_combined.npz"
                )
                np.savez_compressed(outname, waveforms=merged_pmt)
                print(f"✔ PMT {card_id}_{slot_id}_{ch_id}_{pos_id} part{part_id}: {merged_pmt.shape[0]} waveforms saved from {len(chunk_files)} files.")
            else:
                print(f"⚠️ PMT {card_id}_{slot_id}_{ch_id}_{pos_id} part{part_id}: no valid waveforms to merge!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge waveform chunks for PMTs by part")
    parser.add_argument("--folder", required=True, help="Folder containing .npz files")
    parser.add_argument("--pmt-json", required=True, help="JSON file with list of PMTs [card,slot,ch,pos]")
    parser.add_argument("--start", type=int, required=True, help="Start PMT index")
    parser.add_argument("--end", type=int, required=True, help="End PMT index")
    args = parser.parse_args()

    merge_pmts_by_part(args.folder, args.pmt_json, args.start, args.end)
