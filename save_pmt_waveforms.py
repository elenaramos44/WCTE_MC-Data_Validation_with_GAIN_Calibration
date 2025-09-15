import os
import glob
import json
import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

def load_all_parts(run_number, tree_name="WCTEReadoutWindows", max_events=None, base_path=None):
    if base_path is None:
        base_path = "/dipc/elena/WCTE_2025_commissioning/root_files/PMTs_calib"
    
    pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P*.root")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos para patrón: {pattern}")

    arrays = []
    for f in tqdm(files, desc="Loading ROOT files"):
        with uproot.open(f"{f}:{tree_name}") as tree:
            arr = tree.arrays(library="ak", entry_stop=max_events)
            arrays.append(arr)

    concatenated = ak.concatenate(arrays, axis=0)
    return concatenated

def process_and_save(run_number, output_dir, max_events=2000):
    print(f"Loading run {run_number} ...")
    data = load_all_parts(run_number, max_events=max_events)

    print(f"Loaded {len(data)} events")
    waveforms = data["pmt_waveforms"]
    card_ids  = data["pmt_waveform_mpmt_card_ids"]
    slot_ids  = data["pmt_waveform_mpmt_slot_ids"]
    channel_ids = data["pmt_waveform_pmt_channel_ids"]

    waveforms_per_pmt = defaultdict(list)

    for event_waveforms, event_card_ids, event_slot_ids, event_channel_ids in tqdm(    
        zip(waveforms, card_ids, slot_ids, channel_ids),
        desc="Processing events",
        total=len(waveforms)
    ):                                                   #iterates over all events and "event_waveforms" are all PMT waveforms for a particular event
        for wf, cid, sid, chid in zip(event_waveforms, event_card_ids, event_slot_ids, event_channel_ids):
            if sid == -1 or cid > 120:
                continue
            pmt_id = (int(cid), int(sid), int(chid))
            wf_np = np.array(wf)
            baseline = np.mean(wf_np[:10])
            wf_corrected = wf_np - baseline
            waveforms_per_pmt[pmt_id].append(wf_corrected)    #iterates through all PMTs in a particular event + discard bad ones + baseline subtraction

    print(f"Waveforms per PMT found: {len(waveforms_per_pmt)}")   #how many unique PMTs with data where found

    os.makedirs(output_dir, exist_ok=True)

    for pmt_id, wfs in waveforms_per_pmt.items():
        cid, sid, chid = pmt_id
        wfs_array = np.array(wfs)
        
        #save waveforms (already corrected -above-)
        file_path = os.path.join(output_dir, f"card{cid}_slot{sid}_ch{chid}.npz")
        np.savez_compressed(file_path, waveforms=wfs_array)

        
        mean_wf = np.mean(wfs_array, axis=0)
        mean_path = os.path.join(output_dir, f"mean_waveform_card{cid}_slot{sid}_ch{chid}.npy")
        np.save(mean_path, mean_wf)

        print(f"Saved {len(wfs)} waveforms and mean waveform → {file_path}")

    
    metadata = {
        "run_number": run_number,
        "output_directory": output_dir,
        "total_events_processed": int(len(data)),
        "total_pmts_found": len(waveforms_per_pmt),
        "timestamp": datetime.now().isoformat(),
        "selection": {
            "slot_id_valid": "slot_id != -1",
            "card_id_max": "card_id <= 120",
            "baseline_subtraction": "mean of first 10 samples"
        }
    }

    metadata_path = os.path.join(output_dir, f"metadata_run{run_number}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved → {metadata_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="process and save PMT waveforms from ROOT files")
    parser.add_argument("--run", type=int, required=True, help="run_number to process")
    parser.add_argument("--outdir", type=str, required=True, help="output directory where .npz files are stored")
    parser.add_argument("--max_events", type=int, default=2000, help="max_events to be processed")

    args = parser.parse_args()

    process_and_save(args.run, args.outdir, args.max_events)
