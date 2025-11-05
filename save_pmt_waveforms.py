import os
import glob
import json
import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict


def load_all_parts(run_number, tree_name="WCTEReadoutWindows", max_events=None, base_path=None, verbose=False, quiet=False, chunk_size=100):
    if base_path is None:
        base_path = "/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/root_files"

    pattern = os.path.join(base_path, f"WCTE_offline_R{run_number}S0P0.root")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    if verbose:
        print(f"[INFO] Found {len(files)} ROOT files for run {run_number}.")
    elif not quiet:
        print(f"Loading {len(files)} ROOT files...")

    arrays = []
    for f in tqdm(files, desc="Loading ROOT files", disable=quiet):
        print(f"[DEBUG] Opening file: {f}")
        with uproot.open(f"{f}:{tree_name}") as tree:
            n_entries = tree.num_entries
            n_to_read = min(n_entries, max_events) if max_events is not None else n_entries
            print(f"[DEBUG] Tree has {n_entries} entries; reading {n_to_read} entries in chunks of {chunk_size}")
            for start in range(0, n_to_read, chunk_size):
                stop = min(start + chunk_size, n_to_read)
                print(f"[DEBUG] Loading entries {start}–{stop}")
                arr = tree.arrays(
                    [
                        "pmt_waveforms",
                        "pmt_waveform_mpmt_card_ids",
                        "pmt_waveform_mpmt_slot_ids",
                        "pmt_waveform_pmt_channel_ids",
                    ],
                    entry_start=start,
                    entry_stop=stop,
                    library="ak",
                )
                arrays.append(arr)
                print(f"[DEBUG] Finished chunk {start}-{stop}")

    concatenated = ak.concatenate(arrays, axis=0)
    print(f"[INFO] Total concatenated events: {len(concatenated)}")
    return concatenated


def process_and_save(run_number, output_dir, max_events=None, verbose=False, quiet=False):
    if verbose:
        print(f"[INFO] Starting processing for run {run_number}...")
    elif not quiet:
        print(f"Loading run {run_number} ...")

    data = load_all_parts(run_number, max_events=max_events, verbose=verbose, quiet=quiet)

    if not quiet:
        print(f"Loaded {len(data)} events")

    waveforms = data["pmt_waveforms"]
    card_ids  = data["pmt_waveform_mpmt_card_ids"]
    slot_ids  = data["pmt_waveform_mpmt_slot_ids"]
    channel_ids = data["pmt_waveform_pmt_channel_ids"]

    waveforms_per_pmt = defaultdict(list)

    total_pmt_counter = 0

    for event_idx, (event_waveforms, event_card_ids, event_slot_ids, event_channel_ids) in enumerate(
        tqdm(zip(waveforms, card_ids, slot_ids, channel_ids),
             desc="Processing events", total=len(waveforms), disable=quiet)
    ):
        if event_idx % 50 == 0:
            print(f"[DEBUG] Processing event {event_idx}/{len(waveforms)}")

        for wf, cid, sid, chid in zip(event_waveforms, event_card_ids, event_slot_ids, event_channel_ids):
            if sid == -1 or cid > 120:
                #print(f"[WARN] Skipping invalid PMT ID (card={cid}, slot={sid}, ch={chid})")
                continue

            pmt_id = (int(cid), int(sid), int(chid))
            wf_np = np.array(wf)

            if wf_np.size == 0:
                #print(f"[WARN] Empty waveform found for PMT {pmt_id}, skipping")
                continue

            baseline = np.mean(wf_np[:10])
            wf_corrected = wf_np - baseline

            if np.isnan(wf_corrected).any() or np.isinf(wf_corrected).any():
                #print(f"[WARN] NaN or Inf in waveform for PMT {pmt_id}")
                continue

            waveforms_per_pmt[pmt_id].append(wf_corrected)
            total_pmt_counter += 1

            if total_pmt_counter % 100 == 0:
                print(f"[DEBUG] Processed {total_pmt_counter} PMTs so far...")

    if not quiet:
        print(f"[INFO] Total PMTs processed: {total_pmt_counter}")
        print(f"[INFO] Waveforms per PMT found: {len(waveforms_per_pmt)}")

    os.makedirs(output_dir, exist_ok=True)

    for idx, (pmt_id, wfs) in enumerate(waveforms_per_pmt.items()):
        cid, sid, chid = pmt_id
        wfs_array = np.array(wfs)
        file_path = os.path.join(output_dir, f"card{cid}_slot{sid}_ch{chid}.npz")
        np.savez_compressed(file_path, waveforms=wfs_array)
        mean_wf = np.mean(wfs_array, axis=0)
        mean_path = os.path.join(output_dir, f"mean_waveform_card{cid}_slot{sid}_ch{chid}.npy")
        np.save(mean_path, mean_wf)
        if verbose and idx % 50 == 0:
            print(f"[INFO] Saved PMT {idx}: {file_path}")


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

    if not quiet:
        print(f"Metadata saved → {metadata_path}")
    if verbose:
        print(f"[INFO] Processing complete for run {run_number}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="process and save PMT waveforms from ROOT files")
    parser.add_argument("--run", type=int, required=True, help="run_number to process")
    parser.add_argument("--outdir", type=str, required=True, help="output directory where .npz files are stored")
    parser.add_argument("--max_events", type=int, default=None, help="max_events to be processed")
    parser.add_argument("--root-file", type=str, default=None, help="process only this ROOT file")

    # Verbosity options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verbose", action="store_true", help="Enable detailed print messages during processing")
    group.add_argument("--quiet", action="store_true", help="Suppress non-critical messages")

    args = parser.parse_args()

    process_and_save(args.run, args.outdir, args.max_events, verbose=args.verbose, quiet=args.quiet)

