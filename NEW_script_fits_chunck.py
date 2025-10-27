#!/usr/bin/env python3
import numpy as np
import os
import argparse
from scipy.optimize import minimize
from scipy.stats import norm

#----------------- ARGPARSE -----------------
parser = argparse.ArgumentParser(description="Double Gaussian fit for PMTs")
parser.add_argument("--chunk-id", type=int, required=True,
                    help="Index of the PMT chunk to process (0,1,2,...)")
parser.add_argument("--chunk-size", type=int, default=100,
                    help="Number of PMTs per job (default=100)")
args = parser.parse_args()

chunk_id = args.chunk_id
chunk_size = args.chunk_size
start_idx = chunk_id * chunk_size
end_idx   = start_idx + chunk_size
print(f"Processing PMTs {start_idx} to {end_idx-1}")

#----------------- DIRECTORIES -----------------
signal_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2309"

# List all available PMTs
signal_files = [f for f in os.listdir(signal_dir) if f.endswith(".npz")]
pmts_all = sorted([f.replace(".npz","") for f in signal_files])

#----------------- INTEGRATION -----------------
def load_waveforms(npz_file):
    data = np.load(npz_file)
    return data["waveforms"]

def integrate_waveform(wf, pre_peak=5, post_peak=2):
    peak_idx = np.argmax(wf)
    start = max(0, peak_idx - pre_peak)
    end   = min(len(wf), peak_idx + post_peak + 1)
    return np.sum(wf[start:end])

def compute_charges(waveforms):
    return np.array([integrate_waveform(wf) for wf in waveforms])

#----------------- DOUBLE GAUSSIAN FIT -----------------
def stable_nll(params, data):
    mu1, sigma1, mu2, sigma2, w = params
    if sigma1 <= 0 or sigma2 <= 0:
        return 1e300
    w = float(np.clip(w, 1e-9, 1-1e-9))
    lp1 = norm.logpdf(data, loc=mu1, scale=sigma1)
    lp2 = norm.logpdf(data, loc=mu2, scale=sigma2)
    return -np.sum(np.logaddexp(np.log(w)+lp1, np.log(1-w)+lp2))

def fit_double_gauss_multistart(data, n_starts=12):
    best = None
    best_nll = np.inf
    p10, p30, p50, p70, p90 = np.percentile(data, [10,30,50,70,90])
    spe_candidates = data[(data>80) & (data<250)]
    mu2_guess = np.median(spe_candidates) if len(spe_candidates) > 0 else p70

    init_list = []
    mus1 = [0.0, p10, p30]
    mus2 = [mu2_guess, p70, p90]
    sigs = [3.0, 10.0, 20.0]
    ws = [0.7, 0.8, 0.9]
    for mu1 in mus1:
        for mu2 in mus2:
            for s1 in sigs:
                for s2 in sigs:
                    for w in ws:
                        if mu1 < mu2:
                            init_list.append([mu1,s1,mu2,s2,w])
                        if len(init_list) >= n_starts: break
                    if len(init_list) >= n_starts: break
                if len(init_list) >= n_starts: break
            if len(init_list) >= n_starts: break
        if len(init_list) >= n_starts: break

    rng = np.random.default_rng(12345)
    while len(init_list) < n_starts:
        mu1_r = float(rng.normal(0.0, 5.0))
        mu2_r = float(rng.uniform(80, 250))
        s1_r = float(rng.uniform(1.0, 15.0))
        s2_r = float(rng.uniform(5.0, 40.0))
        w_r = float(rng.uniform(0.5, 0.99))
        init_list.append([mu1_r,s1_r,mu2_r,s2_r,w_r])

    bounds = [(-50,50),(0.1,100),(80,250),(0.1,200),(1e-6,1-1e-6)]
    for p0 in init_list:
        res = minimize(stable_nll, p0, args=(data,), method="L-BFGS-B", bounds=bounds)
        if res.success and res.fun < best_nll:
            best_nll = res.fun
            best = res

    if best is None:
        raise RuntimeError("No successful fit found.")

    mu1f, s1f, mu2f, s2f, wf = best.x
    if mu1f > mu2f:
        mu1f, mu2f = mu2f, mu1f
        s1f, s2f = s2f, s1f
        wf = 1.0 - wf
    return {'mu1':mu1f,'sigma1':s1f,'mu2':mu2f,'sigma2':s2f,'w':wf}

#----------------- LOOP OVER PMTs -----------------
results_list = []
failed_pmts = []

for idx, pmt_label in enumerate(pmts_all[start_idx:end_idx], start=start_idx):
    try:
        card_id = int(pmt_label.split("_")[0][4:])
        slot_id = int(pmt_label.split("_")[1][4:])
        channel_id = int(pmt_label.split("_")[2][2:])

        waveforms = load_waveforms(os.path.join(signal_dir, pmt_label + ".npz"))
        charges = compute_charges(waveforms)

        fit = fit_double_gauss_multistart(charges, n_starts=15)
        mu1, s1 = fit['mu1'], fit['sigma1']
        mu2, s2 = fit['mu2'], fit['sigma2']
        w = fit['w']

        gain = mu2 - mu1
        err_gain = np.sqrt(s1**2 + s2**2)

        results_list.append((
            card_id, slot_id, channel_id,
            mu1, s1, len(charges),
            mu2, s2, len(charges),
            gain, err_gain
        ))

        if (idx+1) % 100 == 0:
            print(f"Processed {idx+1} PMTs successfully.")

    except Exception as e:
        failed_pmts.append((pmt_label, str(e)))
        print(f"⚠ Failed PMT: {pmt_label} | Error: {e}")

#----------------- SAVE RESULTS -----------------
dtype = np.dtype([
    ('card_id','i4'),('slot_id','i4'),('channel_id','i4'),
    ('pedestal_mean','f8'),('pedestal_sigma','f8'),('N_pedestal','i4'),
    ('spe_mean','f8'),('spe_sigma','f8'),('N_spe','i4'),
    ('gain','f8'),('gain_error','f8')
])

results_array = np.array(results_list, dtype=dtype)
npz_file_out = f"/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/NEW_doubleGauss_830ns_{chunk_id}.npz"
np.savez(npz_file_out, results=results_array)

if not failed_pmts:
    print(f"✔ Saved results for PMTs {start_idx} to {end_idx-1} without errors → {npz_file_out}")
else:
    print(f"⚠ Saved results for PMTs {start_idx} to {end_idx-1} with {len(failed_pmts)} failures.")
    for f in failed_pmts:
        print(f"   Failed PMT: {f[0]} | Error: {f[1]}")
