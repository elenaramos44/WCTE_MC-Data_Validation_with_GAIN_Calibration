#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse
import glob

def merge_pmts(folder, pmts_json, start_idx, end_idx):
    # Load list of PMTs
    with open(pmts_json, "r") as f:
        pmts = json.load(f)  # list of (card_id, slot_id, ch_id, pos_id)
    
    # Safe slicing
    pmts_to_process = pmts[start_idx : min(end_idx+1, len(pmts))]

    if not pmts_to_process:
        print(f"No PMTs to process in range {start_idx}-{end_idx}.")
        return

    for card_id, slot_id, ch_id, pos_id in pmts_to_process:
        all_waveforms = []

        # Find all parts/chunks for this PMT
        chunk_files = sorted(
            glob.glob(os.path.join(
                folder,
                f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}_part*_chunk*.npz"
            )),
            key=lambda x: int(x.split("_chunk")[1].split(".npz")[0])
        )

        if not chunk_files:
            print(f"⚠️ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: no files found.")
            continue

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
                folder, f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}_combined.npz"
            )
            np.savez_compressed(outname, waveforms=merged_pmt)
            print(f"✔ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: {merged_pmt.shape[0]} waveforms saved from {len(chunk_files)} files.")
        else:
            print(f"⚠️ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: no valid waveforms to merge!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge waveform chunks for PMTs (all parts, all chunks)")
    parser.add_argument("--folder", required=True, help="Folder containing .npz files")
    parser.add_argument("--pmt-json", required=True, help="JSON file with list of PMTs [card,slot,ch,pos]")
    parser.add_argument("--start", type=int, required=True, help="Start PMT index")
    parser.add_argument("--end", type=int, required=True, help="End PMT index")
    args = parser.parse_args()

    merge_pmts(args.folder, args.pmt_json, args.start, args.end)
