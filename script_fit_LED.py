#!/usr/bin/env python3
import sys
sys.path.append("/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration")
import gain_utils as gu
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import argparse
import glob
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize




#----------------- ARGPARSE -----------------
parser = argparse.ArgumentParser(description="Double Gaussian fit for PMTs (parquet)")
parser.add_argument("--chunk-id", type=int, required=True,
                    help="Index of the PMT chunk to process (0,1,2,...)")
parser.add_argument("--chunk-size", type=int, default=100,
                    help="Number of PMTs per job (default=100)")
parser.add_argument("--folder", type=str, required=True,
                    help="Folder with parquet files")
parser.add_argument("--run-number", type=str, required=True,
                    help="Run number identifier for parquet files")
args = parser.parse_args()

chunk_id = args.chunk_id
chunk_size = args.chunk_size
folder = args.folder
run_number = args.run_number

start_idx = chunk_id * chunk_size
end_idx   = start_idx + chunk_size
print(f"Processing PMTs {start_idx} to {end_idx-1}")

#----------------- LOAD DATA -----------------
df_wf = gu.load_waveforms(folder, run_number)
df_led = gu.load_led(folder, run_number)

#----------------- UNIQUE PMTS -----------------
pmts = df_wf[["card_id","chan"]].drop_duplicates().sort_values(["card_id","chan"]).to_numpy()
pmts_chunk = pmts[start_idx:end_idx]

#----------------- INTEGRATION -----------------
threshold = 10  # ADC counts

results_list = []
failed_pmts = []

for card_id, chan in pmts_chunk:
    try:
        df_sel = df_wf[(df_wf['card_id'] == card_id) & (df_wf['chan'] == chan)]
        waveforms = np.stack(df_sel['samples'].to_numpy())
        waveforms_bs = np.array([gu.baseline_subtract(wf) for wf in waveforms])

        #compute CFD times for alignment
        cfd_times = np.array([gu.get_cfd(wf)[0] for wf in waveforms_bs])
        target_bin = 15
        cfd_bins = np.rint(cfd_times).astype(int)

        #align waveforms
        aligned_wfs = np.zeros_like(waveforms_bs)
        for i, wf in enumerate(waveforms_bs):
            shift = target_bin - cfd_bins[i]
            aligned_wfs[i] = np.roll(wf, shift)

        #baseline-subtracted after alignment
        waveforms_bs = np.array([gu.baseline_subtract(wf) for wf in aligned_wfs])

        # compute integrated charges with threshold
        signal_charges = np.array([
            gu.integrate_waveform_control(wf) if np.max(wf) <= threshold
            else gu.integrate_waveform_signal(wf, pre_peak=2, post_peak=1)
            for wf in waveforms_bs
        ])

        #unbinned double Gaussian fit
        fit_results = gu.fit_double_gauss_multistart(signal_charges)
        mu1, sigma1 = fit_results["mu1"], fit_results["sigma1"]
        mu2, sigma2 = fit_results["mu2"], fit_results["sigma2"]
        w = fit_results["w"]

        #results
        results_list.append((
            int(card_id), int(chan),
            mu1, sigma1, len(signal_charges),
            mu2, sigma2, len(signal_charges),
            mu2-mu1,  # gain
            np.sqrt((sigma1/np.sqrt(len(signal_charges)))**2 + (sigma2/np.sqrt(len(signal_charges)))**2)  # gain error
        ))

    except Exception as e:
        failed_pmts.append((f"{card_id}_{chan}", str(e)))
        continue

#----------------- SAVE RESULTS -----------------
dtype = np.dtype([
    ('card_id','i4'),('chan','i4'),
    ('pedestal_mean','f8'),('pedestal_sigma','f8'),('N_pedestal','i4'),
    ('spe_mean','f8'),('spe_sigma','f8'),('N_spe','i4'),
    ('gain','f8'),('gain_error','f8')
])

results_array = np.array(results_list, dtype=dtype)
npz_file_out = f"/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/pmt_charge_fit_LED_parquet_{chunk_id}.npz"
np.savez(npz_file_out, results=results_array)

if not failed_pmts:
    print(f"✔ Saved results for PMTs {start_idx} to {end_idx-1} without errors → {npz_file_out}")
else:
    print(f"⚠ Saved results for PMTs {start_idx} to {end_idx-1} with {len(failed_pmts)} failures.")
    for f in failed_pmts:
        print(f"   Failed PMT: {f[0]} | Error: {f[1]}")
