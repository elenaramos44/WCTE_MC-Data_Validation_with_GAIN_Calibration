import os
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import norm



# ----------------------------
# Gain calibration 
# ----------------------------


#to read .parquet files (LED runs), instead of .root (diffuser laser ball)

def load_parquet_run(folder, run, kind="waveforms", max_rows=None):
    file = Path(folder) / f"{run}_{kind}.parquet"
    pf = ParquetFile(str(file))
    tot_rows = pf.metadata.num_rows
    n_rows = min(tot_rows, max_rows) if max_rows else tot_rows
    batch = next(pf.iter_batches(batch_size=n_rows))
    return pa.Table.from_batches([batch]).to_pandas()

def load_waveforms(folder, run, max_rows=None):
    return load_parquet_run(folder, run, kind="waveforms", max_rows=max_rows)

def load_led(folder, run, max_rows=None):
    return load_parquet_run(folder, run, kind="led", max_rows=max_rows)



def baseline_subtract(wf, n_baseline=10):   #this removes the electronic pedestal 
    baseline = np.mean(wf[:n_baseline])
    return wf - baseline

def integrate_waveform_signal(wf, pre_peak=2, post_peak=1):  #INTEGRATION WINDOW! 
    peak_idx = np.argmax(wf)
    start = max(0, peak_idx - pre_peak)
    end   = min(len(wf), peak_idx + post_peak + 1)
    return np.sum(wf[start:end])

def integrate_waveform_control(wf, window=4):
    peak_idx = np.argmax(wf)
    half_w = window // 2
    start = max(0, peak_idx - half_w)
    end   = min(len(wf), peak_idx + half_w)
    return np.sum(wf[start:end])

def compute_charges_signal(waveforms):
    return np.array([integrate_waveform_signal(wf) for wf in waveforms])

def compute_charges_control(waveforms):
    return np.array([integrate_waveform_control(wf) for wf in waveforms])


#----------------------------------------------------------------
#for consistency!

def charge_calculation_mPMT_method(wf,peak_sample):
    #peak sample is the index of the peak in the waveform
    #this is the value returned by pulse_finding.do_pulse_finding
    charge = np.sum(wf[peak_sample-5:peak_sample+2])
    if wf[peak_sample+2]>0:
        charge+=wf[peak_sample+2]
    return charge

#-----------------------------------------------------------------


def stable_nll(params, data):
    mu1, sigma1, mu2, sigma2, w = params
    if sigma1 <= 0 or sigma2 <= 0:
        return 1e300
    w = float(np.clip(w, 1e-9, 1-1e-9))
    lp1 = norm.logpdf(data, loc=mu1, scale=sigma1)
    lp2 = norm.logpdf(data, loc=mu2, scale=sigma2)
    return -np.sum(np.logaddexp(np.log(w) + lp1, np.log(1-w) + lp2))

def fit_double_gauss_multistart(data, n_starts=12, seed=12345):
    best = None
    best_nll = np.inf
    p10, p30, p50, p70, p90 = np.percentile(data, [10,30,50,70,90])
    spe_candidates = data[(data > 80) & (data < 200)]
    mu2_guess = np.median(spe_candidates) if len(spe_candidates) > 0 else p70
    init_list = []
    mus1 = [0.0, p10, p30]
    mus2 = [mu2_guess, p70, p90]
    sigs = [3.0, 10.0, 20.0]
    ws = [0.1, 0.3, 0.5]
    for mu1 in mus1:
        for mu2 in mus2:
            for s1 in sigs:
                for s2 in sigs:
                    for w in ws:
                        init_list.append([mu1, s1, mu2, s2, w])
                        if len(init_list) >= n_starts:
                            break
    rng = np.random.default_rng(seed)
    while len(init_list) < n_starts:
        init_list.append([
            float(rng.normal(loc=0.0, scale=5.0)),
            float(rng.uniform(1.0, 15.0)),
            float(rng.uniform(p50, p90 + 20)),
            float(rng.uniform(5.0, 40.0)),
            float(rng.uniform(0.01, 0.9))
        ])
    bounds = [(-50,50), (0.1,100), (-10,500), (0.1,200), (1e-6,1-1e-6)]
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
    return {"mu1": mu1f, "sigma1": s1f, "mu2": mu2f, "sigma2": s2f, "w": wf}




# ----------------------
# Timing calibration - this has been directly taken from process_parquet.py 
# ----------------------

cfd_raw_t = [
    0.16323452713658082, 0.20385733509493395, 0.24339187740767365,
    0.2822514122310461, 0.3208335490313887, 0.35953379168152044,
    0.3987592183841288, 0.4389432980060811, 0.4805630068163285,
    0.5241597383052767, 0.5703660640730557, 0.6199413381955754,
    0.6738206794685682, 0.7331844507933303, 0.7995598000823612,
    0.874973724581176, 0.9621917102137131, 1.0301530251726216,
    1.0769047405430523, 1.1210801763323819, 1.1632345271365807
]

amp_raw_t = [
    2.0413475167493225, 2.0642014124776784, 2.0847238089021274,
    2.1028869067818117, 2.118667914530039, 2.1320484585033723,
    2.1430140317025583, 2.151553497195665, 2.1576586607668613,
    2.1613239251470255, 2.162546035746829, 2.1613239251470255,
    2.1576586607668617, 2.1515534971956654, 2.143014031702558,
    2.1320484585033723, 2.118667914530039, 2.1028869067818117,
    2.0847238089021274, 2.0642014124776784, 2.0413475167493225
]

cfd_true_t = [-0.5 + 0.05*i for i in range(21)]

def get_peak_timebins(waveform, threshold, card_id=None):
    values, counts = np.unique(waveform, return_counts=True)
    baseline = values[np.argmax(counts)]
    below = (waveform[0] - baseline) <= threshold
    peak_timebins = []
    max_val = 0
    max_timebin = -1
    for i in range(len(waveform)):
        if below and (waveform[i] - baseline) > threshold:
            below = False
            max_val = 0
            max_timebin = -1
        if not below:
            if (waveform[i] - baseline) > max_val:
                max_val = waveform[i] - baseline
                max_timebin = i
            if (waveform[i] - baseline) <= threshold:
                below = True
                peak_timebins.append(max_timebin)
    return peak_timebins

def get_cfd(adcs):
    c = -2.0
    d = 2
    baseline = (adcs[0] + adcs[1] + adcs[2]) / 3.0   #baseline calculated as the mean of the first 3 samples
    n_largest_vals = sorted(np.array(adcs)-baseline, reverse=True)[:3]
    amp = sum(n_largest_vals)
    data = [(adcs[i]-baseline) + c*(adcs[i-d]-baseline) for i in range(d, len(adcs))]
    max_diff = 0
    i_md = -1
    for iv in range(1, len(data)):
        if data[iv-1] > 0. and data[iv] < 0.:
            if data[iv-1] - data[iv] > max_diff:
                max_diff = data[iv-1] - data[iv]
                i_md = iv
    if i_md > -1:
        x0, y0 = i_md-1, data[i_md-1]
        x1, y1 = i_md, data[i_md]
        x = x0 - (x1-x0)/(y1-y0)*y0
        apply_correction = True
        offset = 5.0
        delta = x - offset
        t = None
        if apply_correction:
            if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                t = offset + correct_t
            elif delta < cfd_raw_t[0]:
                delta += 1
                if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                    correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                    t = offset - 1 + correct_t
            elif delta > cfd_raw_t[-1]:
                delta -= 1
                if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                    correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                    t = offset + 1 + correct_t
        if t is None:
            t = x - 0.5703
            amp = amp / 2.118
        else:
            correct_amp = np.interp(correct_t, cfd_true_t, amp_raw_t)
            amp /= correct_amp
    else:
        t = -999
        amp = -999
    return t, amp, baseline
