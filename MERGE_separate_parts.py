#!/usr/bin/env python3
import os
import numpy as np
import json
import argparse

def merge_pmts_parts(folder, pmts_json, start_idx, end_idx):
    # Load list of PMTs
    with open(pmts_json, "r") as f:
        pmts = json.load(f)  # list of (card_id, slot_id, ch_id, pos_id)
    
    # Safe slicing
    pmts_to_process = pmts[start_idx : min(end_idx+1, len(pmts))]

    if not pmts_to_process:
        print(f"No PMTs to process in range {start_idx}-{end_idx}.")
        return

    failed_pmts = []

    for card_id, slot_id, ch_id, pos_id in pmts_to_process:
        # Check part0 and part1 files
        part_files = []
        missing_parts = []

        for part_id in [0, 1]:
            fpath = os.path.join(
                folder, f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}_part{part_id}_combined.npz"
            )
            if os.path.isfile(fpath):
                part_files.append(fpath)
            else:
                missing_parts.append(part_id)

        if not part_files:
            print(f"❌ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: both parts missing!")
            failed_pmts.append((card_id, slot_id, ch_id, pos_id, "both parts missing"))
            continue
        elif missing_parts:
            print(f"⚠️ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: missing part(s) {missing_parts}, will merge available part(s)")

        # Load available waveforms
        all_waveforms = []
        total_events = 0
        for fpath in part_files:
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
                failed_pmts.append((card_id, slot_id, ch_id, pos_id, f"failed to load {os.path.basename(fpath)}"))

        if all_waveforms:
            merged_waveforms = np.concatenate(all_waveforms, axis=0)
            outname = os.path.join(
                folder, f"card{card_id}_slot{slot_id}_ch{ch_id}_pos{pos_id}_merge.npz"
            )
            np.savez_compressed(outname, waveforms=merged_waveforms)
            print(f"✔ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: {merged_waveforms.shape[0]} waveforms saved in {os.path.basename(outname)}")
        else:
            print(f"❌ PMT {card_id}_{slot_id}_{ch_id}_{pos_id}: no valid waveforms to merge!")
            failed_pmts.append((card_id, slot_id, ch_id, pos_id, "no valid waveforms"))

    # Print summary of failed PMTs
    if failed_pmts:
        print("\n===== FAILED PMTs SUMMARY =====")
        for card_id, slot_id, ch_id, pos_id, reason in failed_pmts:
            print(f"PMT {card_id}_{slot_id}_{ch_id}_{pos_id} -> {reason}")
        print("================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge part0 and part1 combined waveforms for PMTs")
    parser.add_argument("--folder", required=True, help="Folder containing .npz files")
    parser.add_argument("--pmt-json", required=True, help="JSON file with list of PMTs [card,slot,ch,pos]")
    parser.add_argument("--start", type=int, required=True, help="Start PMT index")
    parser.add_argument("--end", type=int, required=True, help="End PMT index")
    args = parser.parse_args()

    merge_pmts_parts(args.folder, args.pmt_json, args.start, args.end)
