#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless para SLURM
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from matplotlib import rcParams

# ----------------------- Matplotlib formatting -----------------------
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['figure.figsize'] = [11, 8]
rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['legend.fontsize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16

# ----------------------- User parameters -----------------------
signal_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/waveforms_including_position"
match_run = re.search(r"run(\d{4})", signal_dir)
run_number = match_run.group(1) if match_run else "unknown"
plots_dir = os.path.join(signal_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ----------------------- Utilities -----------------------
def load_waveforms(npz_file):
    return np.load(npz_file)["waveforms"]

def do_pulse_finding(waveform):
    threshold = 20
    fIntegralPreceding = 4
    fIntegralFollowing = 2
    above_threshold = np.where(waveform[3:-2] > threshold)[0] + 3
    pulses_found = []
    last_index = 0
    for index in above_threshold:
        if waveform[index] <= waveform[index-1]: continue
        if waveform[index] < waveform[index+1]: continue
        if waveform[index] <= waveform[index+2]: continue
        if waveform[index] <= waveform[index-2]: continue
        start = max(0, index - fIntegralPreceding)
        end = min(len(waveform), index + fIntegralFollowing + 1)
        integral = np.sum(waveform[start:end])
        if integral < threshold * 2: continue
        if (last_index > 0) and (index - last_index) <= 20: continue
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

def fit_pedestal_and_spe(charges):
    x = np.asarray(charges)
    if len(x) < 80:
        raise RuntimeError("Too few points to identify peaks")

    xs = np.linspace(np.min(x), np.max(x), 2500)
    kde = gaussian_kde(x)
    ys = kde(xs)
    peak_idx = argrelextrema(ys, np.greater)[0]
    if len(peak_idx) < 2:
        raise RuntimeError("Not enough peaks found")

    expected_spe_min = 100.0
    expected_spe_max = 200.0
    peak_xs = xs[peak_idx]
    peak_ys = ys[peak_idx]
    spe_candidates_mask = (peak_xs >= expected_spe_min) & (peak_xs <= expected_spe_max)
    spe_candidates_idx = np.where(spe_candidates_mask)[0]

    if len(spe_candidates_idx) >= 1:
        chosen_spe_rel = spe_candidates_idx[np.argmax(peak_ys[spe_candidates_idx])]
        spe_center = peak_xs[chosen_spe_rel]
        other_idx = [i for i in range(len(peak_xs)) if i != chosen_spe_rel]
        ped_center = peak_xs[other_idx[np.argmin(peak_xs[other_idx])]]
    else:
        idx_sorted = peak_idx[np.argsort(ys[peak_idx])][-2:]
        idx_sorted = np.sort(idx_sorted)
        ped_center = xs[idx_sorted[0]]
        spe_center = xs[idx_sorted[1]]

    ped_mask = np.abs(x - ped_center) < 6
    if ped_mask.sum() < 20:
        ped_mask = np.abs(x - ped_center) < 10
    ped_data = x[ped_mask]

    sep = abs(spe_center - ped_center)
    spe_hw = max(40, 0.35 * sep)
    spe_mask = np.abs(x - spe_center) < spe_hw
    if spe_mask.sum() < 30:
        spe_mask = np.abs(x - spe_center) < (1.6 * spe_hw)
    spe_data = x[spe_mask]
    if len(spe_data) < 20:
        raise RuntimeError("Not enough points near SPE peak")

    pedestal_fit = fit_gaussian_with_bounds(ped_data, np.median(ped_data), np.std(ped_data) if ped_data.std() > 0 else 1.0, (0.1, 2.0))
    spe_fit = fit_gaussian_with_bounds(spe_data, np.median(spe_data), max(20, np.std(spe_data)), (10, 150))

    gain = spe_fit["mu"] - pedestal_fit["mu"]
    err_gain = np.sqrt(pedestal_fit["err_mu"]**2 + spe_fit["err_mu"]**2)

    return dict(pedestal=pedestal_fit, spe=spe_fit, gain=gain, err_gain=err_gain, n_ped=len(ped_data), n_spe=len(spe_data))

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_and_plot_channel(charges, ax, label):
    x = np.asarray(charges)
    counts, bin_edges = np.histogram(x, bins=500)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

    try:
        seeds = fit_pedestal_and_spe(charges)
        mu_ped, sigma_ped = seeds['pedestal']['mu'], seeds['pedestal']['sigma']
        mu_spe, sigma_spe = seeds['spe']['mu'], seeds['spe']['sigma']

        # Pedestal fit
        ped_mask = (bin_centers >= mu_ped - 3*sigma_ped) & (bin_centers <= mu_ped + 3*sigma_ped)
        if not np.any(ped_mask): ped_mask = bin_centers < 20
        x_fit_ped, y_fit_ped = bin_centers[ped_mask], counts[ped_mask]
        params_ped, _ = curve_fit(gaussian, x_fit_ped, y_fit_ped, p0=[np.max(y_fit_ped), mu_ped, sigma_ped], bounds=([0, mu_ped-5*sigma_ped, 0.05],[np.inf, mu_ped+5*sigma_ped, 5.0]), maxfev=20000)
        A_ped, mu_ped, sigma_ped = params_ped

        # SPE fit
        spe_mask = (bin_centers >= mu_spe - 2*sigma_spe) & (bin_centers <= mu_spe + 2*sigma_spe)
        if not np.any(spe_mask): spe_mask = bin_centers > 20
        x_fit_spe, y_fit_spe = bin_centers[spe_mask], counts[spe_mask]
        params_spe, _ = curve_fit(gaussian, x_fit_spe, y_fit_spe, p0=[np.max(y_fit_spe), mu_spe, sigma_spe], bounds=([0, mu_spe-3*sigma_spe,0.1],[np.inf, mu_spe+3*sigma_spe,80.0]), maxfev=20000)
        A_spe, mu_spe, sigma_spe = params_spe

        # χ²/ndof
        chi2_spe = np.sum((y_fit_spe - gaussian(x_fit_spe, *params_spe))**2 / (y_fit_spe + 1))
        ndof_spe = len(y_fit_spe) - 3
        chi2ndof_spe = chi2_spe / ndof_spe if ndof_spe > 0 else np.nan

        # Plot
        x_full = np.linspace(-20, 500, 2000)
        ax.hist(x, bins=500, histtype='step', color='black', lw=1)
        ax.plot(x_fit_ped, gaussian(x_fit_ped, *params_ped), 'b--', lw=1.2, label=f"Ped μ={mu_ped:.1f}, σ={sigma_ped:.1f}")
        ax.plot(x_fit_spe, gaussian(x_fit_spe, *params_spe), 'g--', lw=1.2, label=f"SPE μ={mu_spe:.1f}, σ={sigma_spe:.1f}")
        ax.plot(x_full, gaussian(x_full, *params_ped)+gaussian(x_full, *params_spe), 'r-', lw=0.8, label="Sum")
        ax.axvspan(x_fit_ped[0], x_fit_ped[-1], color='blue', alpha=0.08)
        ax.axvspan(x_fit_spe[0], x_fit_spe[-1], color='green', alpha=0.08)
        ax.text(0.6,0.65,f"χ²/ndof={chi2ndof_spe:.2f}", transform=ax.transAxes, fontsize=8, color='green')
        ax.set_xlim(-20,500)
        ax.set_ylim(0,300)
        ax.set_title(label, fontsize=8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)
    except Exception:
        ax.hist(x, bins=500, histtype='step', color='black', lw=1)
        ax.set_title(f"{label} (fit failed)", fontsize=8)
        ax.set_xlim(-20,500)
        ax.set_ylim(0,300)
        ax.grid(alpha=0.2)

# ----------------------- Discover mPMTs -----------------------
all_files = os.listdir(signal_dir)
pattern = re.compile(r"card(\d+)_slot(\d+)_ch\d+_pos\d+_merge\.npz$")
available_mpmts = sorted({tuple(map(int, pattern.match(f).groups())) for f in all_files if pattern.match(f)})

# ----------------------- Loop over all mPMTs -----------------------
for card_id, slot_id in available_mpmts:
    fig, axes = plt.subplots(4, 5, figsize=(20,12))
    axes = axes.flatten()

    for ch_id in range(19):
        pmt_label = f"card{card_id}_slot{slot_id}_ch{ch_id}"
        npz_file = next((os.path.join(signal_dir,f) for f in all_files if f.startswith(pmt_label) and "_merge.npz" in f), None)
        if npz_file is None:
            axes[ch_id].axis("off")
            continue

        waveforms = load_waveforms(npz_file)
        peaks = [do_pulse_finding(wf) for wf in waveforms]
        charges = np.array([charge_calculation_mPMT_method(wf, p[0] if len(p)>0 else np.argmax(wf)) for wf,p in zip(waveforms,peaks)])
        fit_and_plot_channel(charges, axes[ch_id], label=pmt_label)

    axes[19].axis("off")
    plt.suptitle(f"Run {run_number} – mPMT card={card_id}, slot={slot_id}", fontsize=14)
    outfile = os.path.join(plots_dir, f"Run{run_number}_card{card_id}_slot{slot_id}.jpg")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
