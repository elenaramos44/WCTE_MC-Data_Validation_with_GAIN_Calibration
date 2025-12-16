#!/usr/bin/env python3
import os
import numpy as np
import argparse
import fnmatch
from scipy.stats import norm, gaussian_kde
from scipy.optimize import minimize, curve_fit
from scipy.signal import argrelextrema

# ----------------- ARGPARSE -----------------
parser = argparse.ArgumentParser(description="Two-step Gaussian fit for PMTs (batch mode)")
parser.add_argument("--pattern", type=str, default="*_merge.npz",
                    help="Pattern to select PMT files, e.g., '*part0_combined.npz'")
parser.add_argument("--chunk-id", type=int, default=0, help="Index of the PMT chunk to process (0,1,2,...)")
parser.add_argument("--chunk-size", type=int, default=100, help="Number of PMTs per job")
args = parser.parse_args()
chunk_id = args.chunk_id
chunk_size = args.chunk_size

# ----------------- DIRECTORIES -----------------
signal_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/waveforms_including_position"

signal_files = [
    f for f in os.listdir(signal_dir)
    if fnmatch.fnmatch(f, args.pattern)
]

pmts_all = sorted([f.replace(".npz", "") for f in signal_files])
start_idx = chunk_id * chunk_size
end_idx = min(start_idx + chunk_size, len(pmts_all))

# ----------------- WCTE FUNCTIONS -----------------
def do_pulse_finding(waveform):
    threshold = 20
    fIntegralPreceding = 4
    fIntegralFollowing = 2
    
    above_threshold = np.where(waveform[3:-2] > threshold)[0] + 3
    pulses_found = []
    last_index = 0
    
    for index in above_threshold:
        if (waveform[index] <= waveform[index-1]): continue
        if (waveform[index] < waveform[index+1]): continue
        if (waveform[index] <= waveform[index+2]): continue
        if (waveform[index] <= waveform[index-2]): continue
        
        start = max(0, index - fIntegralPreceding)
        end = min(len(waveform), index + fIntegralFollowing + 1)
        integral = np.sum(waveform[start:end])
        if integral < threshold * 2:
            continue
        
        if (last_index > 0) and (index - last_index) <= 20:
            continue
        
        pulses_found.append(index)
        last_index = index
    
    return pulses_found


def charge_calculation_mPMT_method(wf, peak_sample):
    n = len(wf)
    start = max(0, peak_sample - 5)
    end = min(n, peak_sample + 2)
    charge = np.sum(wf[start:end])
    if peak_sample + 2 < n and wf[peak_sample + 2] > 0:
        charge += wf[peak_sample + 2]
    return charge

# -----------------------------------------------------------------

def nll_gauss(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    pdf = norm.pdf(data, mu, sigma)
    pdf = np.clip(pdf, 1e-12, None)
    return -np.sum(np.log(pdf))


def fit_gaussian_with_bounds(data, mu0, sigma0, sigma_bounds):
    N = len(data)
    A0 = N

    bounds = [(-np.inf, np.inf), sigma_bounds, (1, np.inf)]

    def nll(params):
        mu, sigma, A = params
        if sigma <= 0 or A <= 0:
            return np.inf
        pdf = A * norm.pdf(data, mu, sigma)
        return -np.sum(np.log(pdf.clip(min=1e-12)))

    res = minimize(nll, [mu0, sigma0, A0], method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError("Gaussian fit failed")

    mu_fit, sigma_fit, A_fit = res.x
    err_mu = sigma_fit / np.sqrt(N)
    err_sigma = sigma_fit / np.sqrt(2*N)

    return dict(mu=mu_fit, sigma=sigma_fit, A=A_fit, err_mu=err_mu, err_sigma=err_sigma, n=N)


# ---------------------------------------------------------
#   FIT STEP 1 - find peaks using KDE with OPTION A
# ---------------------------------------------------------
def fit_pedestal_and_spe(charges, label="PMT"):

    x = np.asarray(charges)
    if len(x) < 20:
        raise RuntimeError("Too few points to identify pedestal + SPE peaks reliably")

    xs = np.linspace(np.min(x), np.max(x), 2500)
    kde = gaussian_kde(x)
    ys = kde(xs)

    peak_idx = argrelextrema(ys, np.greater)[0]
    if len(peak_idx) < 2:
        raise RuntimeError("Only one peak found — cannot separate pedestal and SPE")

    # ---- OPTION A: prefer peak in expected SPE range ----
    expected_spe_min = 100.0   # ADC·ns, adjust if needed
    expected_spe_max = 200.0   # ADC·ns, adjust if needed

    peak_xs = xs[peak_idx]
    peak_ys = ys[peak_idx]

    spe_candidates_mask = (peak_xs >= expected_spe_min) & (peak_xs <= expected_spe_max)
    spe_candidates_idx = np.where(spe_candidates_mask)[0]

    if len(spe_candidates_idx) >= 1:
        chosen_spe_rel = spe_candidates_idx[np.argmax(peak_ys[spe_candidates_idx])]
        spe_center = peak_xs[chosen_spe_rel]
        other_idx = [i for i in range(len(peak_xs)) if i != chosen_spe_rel]
        if len(other_idx) == 0:
            raise RuntimeError("Could not identify pedestal peak after SPE selection.")
        ped_center = peak_xs[other_idx[np.argmin(peak_xs[other_idx])]]
    else:
        # fallback: two strongest peaks
        idx_sorted = peak_idx[np.argsort(ys[peak_idx])][-2:]
        idx_sorted = np.sort(idx_sorted)
        ped_center = xs[idx_sorted[0]]
        spe_center = xs[idx_sorted[1]]

    # pedestal window
    ped_mask = np.abs(x - ped_center) < 6
    if ped_mask.sum() < 20:
        ped_mask = np.abs(x - ped_center) < 10
    ped_data = x[ped_mask]

    # SPE window
    sep = abs(spe_center - ped_center)
    spe_hw = max(40, 0.35 * sep)
    spe_mask = np.abs(x - spe_center) < spe_hw
    if spe_mask.sum() < 30:
        spe_mask = np.abs(x - spe_center) < (1.6 * spe_hw)
    spe_data = x[spe_mask]

    if len(spe_data) < 20:
        raise RuntimeError("Not enough points near the SPE peak for a stable fit.")

    pedestal_fit = fit_gaussian_with_bounds(
        ped_data,
        np.median(ped_data),
        np.std(ped_data) if ped_data.std() > 0 else 1.0,
        (0.1, 2.0)
    )

    spe_fit = fit_gaussian_with_bounds(
        spe_data,
        np.median(spe_data),
        max(20, np.std(spe_data)),
        (10, 150)
    )

    gain = spe_fit["mu"] - pedestal_fit["mu"]
    err_gain = np.sqrt(pedestal_fit["err_mu"]**2 + spe_fit["err_mu"]**2)

    result = dict(
        pedestal=pedestal_fit,
        spe=spe_fit,
        gain=gain,
        err_gain=err_gain,
        n_ped=len(ped_data),
        n_spe=len(spe_data)
    )

    return result

# -------------------------------
# FIT STEP 2 remains unchanged
# -------------------------------
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_and_get_gaussians(charges, seeds, bins=500):
    x = np.asarray(charges)

    counts, bin_edges = np.histogram(x, bins=bins)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # ------------------ Pedestal fit ------------------
    mu_seed = seeds['pedestal']['mu']
    sigma_seed = seeds['pedestal']['sigma']
    n_seed  = seeds['pedestal']['n']

    ped_mask = (bin_centers >= mu_seed - 3*sigma_seed) & (bin_centers <= mu_seed + 3*sigma_seed)
    x_fit_ped = bin_centers[ped_mask]
    y_fit_ped = counts[ped_mask]

    p0_ped = [n_seed, mu_seed, sigma_seed]
    bounds_ped = ([0, mu_seed - 5*sigma_seed, 0.05], [np.inf, mu_seed + 5*sigma_seed, 5.0])
    params_ped, _ = curve_fit(gaussian, x_fit_ped, y_fit_ped, p0=p0_ped, bounds=bounds_ped, maxfev=20000)
    A_ped, mu_ped, sigma_ped = params_ped

    # ------------------ SPE fit ------------------
    mu_spe_seed = seeds['spe']['mu']
    sigma_spe_seed = seeds['spe']['sigma']
    n_spe_seed = seeds['spe']['n']

    spe_mask = (bin_centers >= mu_spe_seed - 2*sigma_spe_seed) & (bin_centers <= mu_spe_seed + 2*sigma_spe_seed)
    x_fit_spe = bin_centers[spe_mask]
    y_fit_spe = counts[spe_mask]

    p0_spe = [n_spe_seed, mu_spe_seed, sigma_spe_seed]
    bounds_spe = ([0, mu_spe_seed - 3*sigma_spe_seed, 0.1], [np.inf, mu_spe_seed + 3*sigma_spe_seed, 80.0])
    params_spe, _ = curve_fit(gaussian, x_fit_spe, y_fit_spe, p0=p0_spe, bounds=bounds_spe, maxfev=20000)
    A_spe, mu_spe, sigma_spe = params_spe

    # ------------------ Gain and errors ------------------
    gain = mu_spe - mu_ped
    N_ped = len(x[(x >= mu_seed - 3*sigma_seed) & (x <= mu_seed + 3*sigma_seed)])
    N_spe = len(x[(x >= mu_spe_seed - 2*sigma_spe_seed) & (x <= mu_spe_seed + 2*sigma_spe_seed)])
    err_gain = np.sqrt(sigma_ped**2 / N_ped + sigma_spe**2 / N_spe)

    # ------------------ χ² / ndof SPE ------------------
    chi2_spe = np.sum((y_fit_spe - gaussian(x_fit_spe, *params_spe))**2 / (y_fit_spe + 1))
    ndof_spe = len(y_fit_spe) - 3
    chi2ndof_spe = chi2_spe / ndof_spe if ndof_spe > 0 else np.nan


    return dict(
        pedestal=(mu_ped, sigma_ped, A_ped),
        spe=(mu_spe, sigma_spe, A_spe),
        gain=(gain, err_gain),
        chi2_spe=chi2_spe,
        chi2ndof_spe=chi2ndof_spe
    )


# ----------------- PROCESS PMTs -----------------
results_list = []
failed_pmts = []

for idx, pmt_label in enumerate(pmts_all[start_idx:end_idx], start=start_idx):
    try:
        parts = pmt_label.split("_")

        card_id    = int(parts[0].replace("card", ""))
        slot_id    = int(parts[1].replace("slot", ""))
        channel_id = int(parts[2].replace("ch", ""))
        pos_id     = int(parts[3].replace("pos", ""))

        signal_npz = os.path.join(signal_dir, pmt_label + ".npz")
        data = np.load(signal_npz)
        signal_waveforms = data["waveforms"]

        all_peaks = [do_pulse_finding(wf) for wf in signal_waveforms]
        pulse_mask = np.array([len(p) > 0 for p in all_peaks])

        charges = np.array([
            charge_calculation_mPMT_method(wf, (p[0] if len(p) > 0 else int(np.argmax(wf))))
            for wf, p in zip(signal_waveforms, all_peaks)
        ])

        pulse_count = np.sum(pulse_mask)
        total_waveforms = len(signal_waveforms)
        pulse_ratio = pulse_count / total_waveforms if total_waveforms > 0 else np.nan
        mu_pe = -np.log(1 - pulse_ratio) if pulse_ratio < 1 else np.nan

        seeds = fit_pedestal_and_spe(charges, label=f"PMT {pmt_label}")
        results = fit_and_get_gaussians(charges, seeds, bins=500)

        results_list.append((
            card_id, slot_id, channel_id, pos_id,
            results['pedestal'][0], results['pedestal'][1], seeds['n_ped'],
            results['spe'][0], results['spe'][1], seeds['n_spe'],
            results['gain'][0], results['gain'][1],
            pulse_ratio, mu_pe,
            results['chi2ndof_spe']
        ))

    except Exception as e:
        failed_pmts.append((pmt_label, str(e)))
        continue

# ----------------- SAVE RESULTS -----------------
dtype = np.dtype([
    ('card_id','i4'),('slot_id','i4'),('channel_id','i4'), ('pos_id','i4'),
    ('pedestal_mean','f8'),('pedestal_sigma','f8'),('N_pedestal','i4'),
    ('spe_mean','f8'),('spe_sigma','f8'),('N_spe','i4'),
    ('gain','f8'),('gain_error','f8'),
    ('pulse_ratio','f8'),('mu_pe','f8'),
    ('chi2ndof_spe','f8')
])
results_array = np.array(results_list, dtype=dtype)
out_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration"
os.makedirs(out_dir, exist_ok=True)
np.savez(os.path.join(out_dir, f"High_statistics_run2307_chunk{chunk_id}.npz"), results=results_array)

print(f"Done. Processed PMTs {start_idx}..{end_idx-1}. Failed PMTs: {len(failed_pmts)}")
