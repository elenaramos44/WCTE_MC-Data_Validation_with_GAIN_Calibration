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
def nll_double_gauss(params, data):
    mu1, sigma1, mu2, sigma2, w = params
    if sigma1 <= 0 or sigma2 <= 0 or not (0 < w < 1):
        return np.inf
    pdf = w * norm.pdf(data, mu1, sigma1) + (1 - w) * norm.pdf(data, mu2, sigma2)
    pdf = np.clip(pdf, 1e-12, None)  # avoid log(0)
    return -np.sum(np.log(pdf))


def fit_double_gauss_unbinned(data, mu2_hint=None, n_starts=25, pmt_label="PMT"):
    """
    Robust unbinned double-Gaussian fit for charge distributions.
    Automatically adapts to data range and avoids runaway SPE fits.
    """
    data = np.asarray(data)
    if len(data) < 30:
        raise RuntimeError("Muy pocos puntos para ajustar una doble gaussiana")

    # --- Basic data descriptors ---
    q16, q50, q84 = np.percentile(data, [16, 50, 84])
    q90, q99 = np.percentile(data, [90, 99])
    qmin, qmax = np.min(data), np.max(data)

    # --- Pedestal estimate ---
    ped_mask = data < q50
    mu_ped_guess = np.mean(data[ped_mask]) if np.any(ped_mask) else np.mean(data)
    sigma_ped_guess = np.std(data[ped_mask]) if np.any(ped_mask) else np.std(data)

    # --- SPE initial guess ---
    if mu2_hint is None:
        mu2_hint = q90  # upper part of the distribution
    sigma2_hint = max(10.0, (q99 - q90) / 2)

    # --- Adaptive, physical bounds ---
    bounds = [
        (mu_ped_guess - 5 * sigma_ped_guess, mu_ped_guess + 5 * sigma_ped_guess),  # μ1
        (0.1, max(5.0, sigma_ped_guess * 3)),                                      # σ1
        (0.5 * mu2_hint, 1.5 * mu2_hint),                                          # μ2
        (5.0, max(50.0, q99 - q16)),                                               # σ2
        (0.05, 0.95)                                                               # w
    ]

    # --- Generate multi-start grid ---
    mus1 = np.linspace(mu_ped_guess - 1, mu_ped_guess + 1, 3)
    sigs1 = np.linspace(sigma_ped_guess * 0.5, sigma_ped_guess * 1.5, 3)
    mus2 = np.linspace(mu2_hint - 100, mu2_hint + 100, 3)
    sigs2 = np.linspace(sigma2_hint * 0.5, sigma2_hint * 1.5, 3)
    ws = [0.1, 0.2, 0.3]

    init_list = []
    for mu1 in mus1:
        for s1 in sigs1:
            for mu2 in mus2:
                for s2 in sigs2:
                    for w in ws:
                        if mu1 < mu2:
                            init_list.append([mu1, s1, mu2, s2, w])
                        if len(init_list) >= n_starts:
                            break
                    if len(init_list) >= n_starts: break
                if len(init_list) >= n_starts: break
            if len(init_list) >= n_starts: break
        if len(init_list) >= n_starts: break

    # --- Optimization loop ---
    best = None
    best_nll = np.inf
    for p0 in init_list:
        try:
            res = minimize(nll_double_gauss, p0, args=(data,),
                           method="L-BFGS-B", bounds=bounds)
            if res.success and res.fun < best_nll:
                best = res
                best_nll = res.fun
        except Exception:
            continue

    if best is None:
        raise RuntimeError("No fit found (likely unphysical initial guesses)")

    mu1, s1, mu2, s2, w = best.x

    # --- Ensure ordering (μ1 < μ2) ---
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        s1, s2 = s2, s1
        w = 1 - w

    # --- Sanity check: reject runaway fits ---
    qmin, qmax = np.min(data), np.max(data)
    if mu2 > qmax * 1.5 or s2 > (qmax - qmin):
        print(f"⚠️ Warning: SPE Gaussian escaped fit range — resetting to 1-Gaussian fit region.")
        mu2, s2 = mu2_hint, sigma2_hint
        w = 0.1

    # --- Derived quantities ---
    gain = mu2 - mu1
    err_gain = np.sqrt(s1**2 + s2**2)
    n_neg = np.sum(data < 0)

    # --- Pretty print ---
    print(f"----- Unbinned double Gaussian fit for {pmt_label} -----")
    print(f"Pedestal (μ₁, σ₁) = ({mu1:.2f}, {s1:.2f}) ADC·ns")
    print(f"SPE       (μ₂, σ₂) = ({mu2:.2f}, {s2:.2f}) ADC·ns")
    print(f"Weight w = {w:.3f}")
    print(f"Gain (μ₂ - μ₁) = {gain:.2f} ± {err_gain:.2f} ADC·ns")
    print(f"Negative charges: {n_neg}/{len(data)} ({100*n_neg/len(data):.2f}%)")

    return {
        'mu1': mu1, 'sigma1': s1,
        'mu2': mu2, 'sigma2': s2,
        'w': w, 'gain': gain, 'err_gain': err_gain,
        'nll': best_nll
    }

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

        fit = fit_double_gauss_unbinned(charges, n_starts=25, pmt_label=pmt_label)
        mu1, s1 = fit['mu1'], fit['sigma1']
        mu2, s2 = fit['mu2'], fit['sigma2']
        w = fit['w']
        gain, err_gain = fit['gain'], fit['err_gain']

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
npz_file_out = f"/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/NEW_doubleGauss_run2309_{chunk_id}.npz"
np.savez(npz_file_out, results=results_array)

if not failed_pmts:
    print(f"✔ Saved results for PMTs {start_idx} to {end_idx-1} without errors → {npz_file_out}")
else:
    print(f"⚠ Saved results for PMTs {start_idx} to {end_idx-1} with {len(failed_pmts)} failures.")
    for f in failed_pmts:
        print(f"   Failed PMT: {f[0]} | Error: {f[1]}")
