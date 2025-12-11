#!/usr/bin/env python3
import os
import numpy as np
import awkward as ak
import uproot
from collections import defaultdict
import argparse

# -------------------------------------------------------------------
# Load only a chunk of a single ROOT part
# -------------------------------------------------------------------
def load_file_chunk(run_number, part, chunk_id, chunk_size, tree_name="WCTEReadoutWindows", base_path=None):
    if base_path is None:
        base_path = "/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/root_files"

    f = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P{part}.root")
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

    start = chunk_id * chunk_size
    stop = start + chunk_size

    with uproot.open(f"{f}:{tree_name}") as tree:
        arrays = tree.arrays([
            "pmt_waveforms",
            "pmt_waveform_mpmt_card_ids",
            "pmt_waveform_mpmt_slot_ids",
            "pmt_waveform_pmt_channel_ids"
        ], library="ak", entry_start=start, entry_stop=stop)

    return arrays

# -------------------------------------------------------------------
# Process the chunk already loaded in memory
# -------------------------------------------------------------------
def process_chunk(arrays, part):
    waveforms_per_pmt = defaultdict(list)

    waveforms = arrays["pmt_waveforms"]
    card_ids = arrays["pmt_waveform_mpmt_card_ids"]
    slot_ids = arrays["pmt_waveform_mpmt_slot_ids"]
    channel_ids = arrays["pmt_waveform_pmt_channel_ids"]

    for event_wfs, event_cids, event_sids, event_chids in zip(
        waveforms, card_ids, slot_ids, channel_ids
    ):
        for wf, cid, sid, chid in zip(event_wfs, event_cids, event_sids, event_chids):
            if sid == -1 or cid > 120:
                continue

            # Use part number in the key
            pmt_key = (int(cid), int(sid), int(chid), int(part))

            wf_np = np.array(wf)
            if wf_np.size == 0:
                continue

            baseline = np.mean(wf_np[:10])
            wf_corrected = wf_np - baseline
            if np.isnan(wf_corrected).any():
                continue

            waveforms_per_pmt[pmt_key].append(wf_corrected)

    return waveforms_per_pmt

# -------------------------------------------------------------------
# Save output
# -------------------------------------------------------------------
def save_output(output_dir, chunk_id, waveforms_per_pmt):
    os.makedirs(output_dir, exist_ok=True)
    for (cid, sid, chid, part), wfs in waveforms_per_pmt.items():
        outname = f"card{cid}_slot{sid}_ch{chid}_part{part}_chunk{chunk_id}.npz"
        outpath = os.path.join(output_dir, outname)
        np.savez_compressed(outpath, waveforms=np.array(wfs))

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save PMT waveforms from a single ROOT part in chunks")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    arrays = load_file_chunk(args.run, args.part, args.chunk_id, args.chunk_size)
    waveforms_per_pmt = process_chunk(arrays, args.part)
    save_output(args.outdir, args.chunk_id, waveforms_per_pmt)
