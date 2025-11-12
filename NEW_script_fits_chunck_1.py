import numpy as np
import os
import argparse
from scipy.optimize import minimize
from scipy.stats import norm

#----------------- ARGPARSE -----------------
parser = argparse.ArgumentParser(description="Double Gaussian fit for PMTs with pulse info")
parser.add_argument("--chunk-id", type=int, default=0,
                    help="Index of the PMT chunk to process (0,1,2,...)")
parser.add_argument("--chunk-size", type=int, default=100,
                    help="Number of PMTs per job (default=100)")
args = parser.parse_args()

chunk_id = args.chunk_id
chunk_size = args.chunk_size

#----------------- DIRECTORIES -----------------
signal_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307"
signal_files = [f for f in os.listdir(signal_dir) if f.endswith(".npz")]
pmts_all = sorted([f.replace(".npz","") for f in signal_files])

start_idx = chunk_id * chunk_size
end_idx = min(start_idx + chunk_size, len(pmts_all))

#----------------- FUNCTIONS -----------------
def load_waveforms(npz_file):
    data = np.load(npz_file)
    return data["waveforms"]

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

def nll_double_gauss(params, data):
    mu1, sigma1, mu2, sigma2, w = params
    if sigma1 <= 0 or sigma2 <= 0 or not (0 < w < 1):
        return np.inf
    pdf = w * norm.pdf(data, mu1, sigma1) + (1 - w) * norm.pdf(data, mu2, sigma2)
    pdf = np.clip(pdf, 1e-12, None)
    return -np.sum(np.log(pdf))

def fit_double_gauss_unbinned_robust(charges, n_starts=30):
    data = np.asarray(charges)
    if len(data) < 30:
        raise RuntimeError("Too few points for double Gaussian fit")
    
    ped_mask = data < np.percentile(data, 50)
    mu1_guess = 0.0
    sigma1_guess = np.std(data[ped_mask]) if np.any(ped_mask) else np.std(data)
    
    spe_mask = (data >= 130) & (data <= 180)
    if np.any(spe_mask):
        mu2_guess = np.median(data[spe_mask])
        sigma2_guess = np.std(data[spe_mask])
    else:
        mu2_guess = 150.0
        sigma2_guess = 20.0

    bounds = [(-2, 2), (0.01, 5), (130, 200), (5, 150), (0.01, 0.99)]
    
    mus1 = [mu1_guess]
    sigs1 = [sigma1_guess]
    mus2 = np.linspace(130, 180, 3)
    sigs2 = np.linspace(max(5, sigma2_guess*0.5), sigma2_guess*1.5, 3)
    ws = [0.1, 0.2, 0.3]

    init_list = []
    for mu1 in mus1:
        for s1 in sigs1:
            for mu2 in mus2:
                for s2 in sigs2:
                    for w in ws:
                        init_list.append([mu1, s1, mu2, s2, w])
                        if len(init_list) >= n_starts: break
                    if len(init_list) >= n_starts: break
                if len(init_list) >= n_starts: break
            if len(init_list) >= n_starts: break
        if len(init_list) >= n_starts: break

    best = None
    best_nll = np.inf
    for p0 in init_list:
        try:
            res = minimize(nll_double_gauss, p0, args=(data,), method="L-BFGS-B", bounds=bounds)
            if res.success and res.fun < best_nll:
                best = res
                best_nll = res.fun
        except Exception:
            continue

    if best is None:
        raise RuntimeError("No fit found")

    mu1, sigma1, mu2, sigma2, w = best.x
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        sigma1, sigma2 = sigma2, sigma1
        w = 1 - w

    gain = mu2 - mu1
    err_gain = np.sqrt(sigma1**2 + sigma2**2)
    n_neg = np.sum(data < 0)

    return {"mu1": mu1, "sigma1": sigma1, "mu2": mu2, "sigma2": sigma2,
            "w": w, "gain": gain, "err_gain": err_gain, "nll": best_nll}

#----------------- PROCESS PMTs -----------------
results_list = []
failed_pmts = []

for idx, pmt_label in enumerate(pmts_all[start_idx:end_idx], start=start_idx):
    try:
        card_id = int(pmt_label.split("_")[0][4:])
        slot_id = int(pmt_label.split("_")[1][4:])
        channel_id = int(pmt_label.split("_")[2][2:])
        signal_npz = os.path.join(signal_dir, pmt_label + ".npz")
        signal_waveforms = load_waveforms(signal_npz)

        all_peaks = [do_pulse_finding(wf) for wf in signal_waveforms]
        pulse_mask = np.array([len(p) > 0 for p in all_peaks])
        noise_mask = ~pulse_mask
        charges = []
        peak_values = []

        for wf, peaks in zip(signal_waveforms, all_peaks):
            if len(peaks) > 0:
                peak_sample = peaks[0]
            else:
                peak_sample = np.argmax(wf)
            q = charge_calculation_mPMT_method(wf, peak_sample)
            charges.append(q)
            peak_values.append(wf[peak_sample])

        charges = np.array(charges)
        peak_values = np.array(peak_values)
        pulse_count = np.sum(pulse_mask)
        total_waveforms = len(signal_waveforms)
        pulse_ratio = pulse_count / total_waveforms
        mu_pe = -np.log(1 - pulse_ratio) if pulse_ratio < 1 else np.nan

        fit = fit_double_gauss_unbinned_robust(charges)

        results_list.append((
            card_id, slot_id, channel_id,
            fit['mu1'], fit['sigma1'], np.sum(noise_mask),
            fit['mu2'], fit['sigma2'], pulse_count,
            fit['gain'], fit['err_gain'],
            pulse_ratio, mu_pe
        ))

    except Exception as e:
        failed_pmts.append((pmt_label, str(e)))

#----------------- SAVE RESULTS -----------------
dtype = np.dtype([
    ('card_id','i4'),('slot_id','i4'),('channel_id','i4'),
    ('pedestal_mean','f8'),('pedestal_sigma','f8'),('N_pedestal','i4'),
    ('spe_mean','f8'),('spe_sigma','f8'),('N_spe','i4'),
    ('gain','f8'),('gain_error','f8'),
    ('pulse_ratio','f8'),('mu_pe','f8')
])

results_array = np.array(results_list, dtype=dtype)
npz_file_out = f"/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/NEW_doubleGauss_run2307_v2_chunk{chunk_id}.npz"
np.savez(npz_file_out, results=results_array)
