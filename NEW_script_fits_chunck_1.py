#!/usr/bin/env python3
import numpy as np
import os
import argparse
from scipy.optimize import minimize
from scipy.stats import norm, gaussian_kde
from scipy.signal import argrelextrema

# ----------------- ARGPARSE -----------------
parser = argparse.ArgumentParser(description="Double Gaussian fit for PMTs with pulse info (batch mode)")
parser.add_argument("--chunk-id", type=int, default=0, help="Index of the PMT chunk to process (0,1,2,...)")
parser.add_argument("--chunk-size", type=int, default=100, help="Number of PMTs per job")
args = parser.parse_args()
chunk_id = args.chunk_id
chunk_size = args.chunk_size

# ----------------- DIRECTORIES -----------------
signal_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307"
signal_files = [f for f in os.listdir(signal_dir) if f.endswith(".npz")]
pmts_all = sorted([f.replace(".npz","") for f in signal_files])
start_idx = chunk_id * chunk_size
end_idx = min(start_idx + chunk_size, len(pmts_all))

# ----------------- FUNCTIONS -----------------
def do_pulse_finding(wf):
    threshold = 20
    fIntegralPreceding = 4
    fIntegralFollowing = 2
    above_threshold = np.where(wf[3:-2]>threshold)[0]+3
    pulses = []
    last_index = 0
    for idx in above_threshold:
        if wf[idx]<=wf[idx-1] or wf[idx]<wf[idx+1] or wf[idx]<=wf[idx+2] or wf[idx]<=wf[idx-2]:
            continue
        start = max(0, idx-fIntegralPreceding)
        end = min(len(wf), idx+fIntegralFollowing+1)
        integral = np.sum(wf[start:end])
        if integral < threshold*2 or (last_index>0 and idx-last_index<=20):
            continue
        pulses.append(idx)
        last_index = idx
    return pulses

def charge_calculation_mPMT_method(wf, peak_sample):
    start = max(0, peak_sample-5)
    end = min(len(wf), peak_sample+2)
    charge = np.sum(wf[start:end])
    if peak_sample+2 < len(wf) and wf[peak_sample+2]>0:
        charge += wf[peak_sample+2]
    return charge

def nll_gauss(params, data):
    mu, sigma = params
    if sigma<=0: return np.inf
    pdf = np.clip(norm.pdf(data, mu, sigma), 1e-12, None)
    return -np.sum(np.log(pdf))

def fit_gaussian_with_bounds(data, mu0, sigma0, sigma_bounds):
    bounds = [(-np.inf, np.inf), sigma_bounds]
    res = minimize(nll_gauss, [mu0, sigma0], args=(data,), method="L-BFGS-B", bounds=bounds)
    if not res.success: raise RuntimeError("Gaussian fit failed")
    mu_fit, sigma_fit = res.x
    N = len(data)
    return dict(mu=mu_fit, sigma=sigma_fit, err_mu=sigma_fit/np.sqrt(N), err_sigma=sigma_fit/np.sqrt(2*N), n=N, nll=res.fun)

def fit_pedestal_and_spe(charges):
    x = np.asarray(charges)
    if len(x)<80: raise RuntimeError("Too few points for pedestal + SPE")
    xs = np.linspace(np.min(x), np.max(x), 2500)
    kde = gaussian_kde(x)
    ys = kde(xs)
    peak_idx = argrelextrema(ys, np.greater)[0]
    if len(peak_idx)<2: raise RuntimeError("Cannot separate pedestal and SPE")
    idx_sorted = np.sort(peak_idx[np.argsort(ys[peak_idx])][-2:])
    ped_center, spe_center = xs[idx_sorted[0]], xs[idx_sorted[1]]
    ped_mask = np.abs(x-ped_center)<6
    if ped_mask.sum()<20: ped_mask = np.abs(x-ped_center)<10
    spe_mask = np.abs(x-spe_center)<max(40,0.35*abs(spe_center-ped_center))
    if spe_mask.sum()<30: spe_mask = np.abs(x-spe_center)<(1.6*max(40,0.35*abs(spe_center-ped_center)))
    ped_data = x[ped_mask]; spe_data = x[spe_mask]
    if len(spe_data)<20: raise RuntimeError("Not enough points near SPE peak")
    pedestal_fit = fit_gaussian_with_bounds(ped_data, mu0=np.median(ped_data), sigma0=max(np.std(ped_data),1e-3), sigma_bounds=(0.1,2.0))
    spe_fit = fit_gaussian_with_bounds(spe_data, mu0=np.median(spe_data), sigma0=max(20,np.std(spe_data)), sigma_bounds=(10,150))
    return dict(pedestal=pedestal_fit, spe=spe_fit, n_ped=len(ped_data), n_spe=len(spe_data))

def nll_double_gauss(params,data):
    mu1,s1,mu2,s2,w = params
    if s1<=0 or s2<=0 or not (0<w<1): return np.inf
    pdf = np.clip(w*norm.pdf(data,mu1,s1)+(1-w)*norm.pdf(data,mu2,s2),1e-12,None)
    return -np.sum(np.log(pdf))

def fit_double_gauss_unbinned_from_separate(charges,res_sep,n_starts=40,rng_seed=1234):
    data = np.asarray(charges)
    mu1_seed = res_sep["pedestal"]["mu"]; s1_seed=res_sep["pedestal"]["sigma"]
    mu2_seed = res_sep["spe"]["mu"]; s2_seed=res_sep["spe"]["sigma"]
    w_seed = res_sep["n_ped"]/(res_sep["n_ped"]+res_sep["n_spe"])
    bounds = [(mu1_seed-10,mu1_seed+10),(0.05,4.0),(mu2_seed-80,mu2_seed+80),(s2_seed-40,s2_seed+40),(0.01,0.99)]
    rng = np.random.default_rng(rng_seed)
    init_list = [[mu1_seed,s1_seed,mu2_seed,s2_seed,w_seed]] + [
        [mu1_seed+rng.normal(0,0.2), max(0.1,s1_seed+rng.normal(0,0.1*s1_seed)), mu2_seed+rng.normal(0,1.5), max(10,s2_seed+rng.normal(0,0.15*s2_seed)), np.clip(w_seed+rng.normal(0,0.03),0.05,0.95)]
        for _ in range(n_starts-1)
    ]
    best, best_nll = None, np.inf
    for p0 in init_list:
        try:
            res = minimize(nll_double_gauss, p0, args=(data,), method="L-BFGS-B", bounds=bounds, options={'maxiter':20000})
            if res.success and res.fun<best_nll:
                best, best_nll=res,res.fun
        except: continue
    if best is None: raise RuntimeError("Double Gaussian fit failed")
    mu1,s1,mu2,s2,w = best.x
    if mu1>mu2: mu1,mu2 = mu2,mu1; s1,s2=s2,s1; w=1-w
    gain = mu2-mu1; err_gain=np.sqrt(s1**2+s2**2)
    return dict(mu1=mu1,sigma1=s1,mu2=mu2,sigma2=s2,w=w,gain=gain,err_gain=err_gain,nll=best_nll)

# ----------------- PROCESS PMTs -----------------
results_list=[]; failed_pmts=[]
for idx,pmt_label in enumerate(pmts_all[start_idx:end_idx],start=start_idx):
    try:
        card_id = int(pmt_label.split("_")[0][4:])
        slot_id = int(pmt_label.split("_")[1][4:])
        channel_id = int(pmt_label.split("_")[2][2:])
        signal_npz = os.path.join(signal_dir,pmt_label+".npz")
        data = np.load(signal_npz)
        signal_waveforms = data["waveforms"]

        all_peaks = [do_pulse_finding(wf) for wf in signal_waveforms]
        pulse_mask = np.array([len(p)>0 for p in all_peaks])
        noise_mask = ~pulse_mask

        charges = np.array([charge_calculation_mPMT_method(wf,(p[0] if len(p)>0 else int(np.argmax(wf)))) for wf,p in zip(signal_waveforms,all_peaks)])

        pulse_count = np.sum(pulse_mask)
        total_waveforms = len(signal_waveforms)
        pulse_ratio = pulse_count/total_waveforms if total_waveforms>0 else np.nan
        mu_pe = -np.log(1-pulse_ratio) if pulse_ratio<1 else np.nan

        res_sep = fit_pedestal_and_spe(charges)
        fit = fit_double_gauss_unbinned_from_separate(charges,res_sep,n_starts=40)
        
        # -----------------------------
        # Compute SPE chi² as in notebook
        # -----------------------------

    #------------------------------------------------------------------------------------------------------------------
        x = np.asarray(charges)
        mu2, sigma2 = fit_dg["mu2"], fit_dg["sigma2"]
        w = fit_dg["w"]

        spe_mask = (x > mu2 - 1.5*sigma2) & (x < mu2 + 1.5*sigma2)
        x_spe = x[spe_mask]

        bins = np.linspace(x_spe.min(), x_spe.max(), 50)
        hist_counts, bin_edges = np.histogram(x_spe, bins=bins)
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]

        # Expected counts from SPE Gaussian only (not pedestal)
        expected_counts = (1 - w) * norm.pdf(bin_centers, mu2, sigma2) * len(x) * bin_width

        # Avoid zero counts in expected (to prevent division by zero)
        expected_counts = np.clip(expected_counts, 1e-6, None)

        # Chi²
        chi2 = np.sum((hist_counts - expected_counts)**2 / expected_counts)
        ndof = len(hist_counts) - 2  # μ2 and sigma2 fitted locally

#------------------------------------------------------------------------------------------------------------------

        x = charges
        mu2, sigma2, w = fit["mu2"], fit["sigma2"], fit["w"]
        spe_mask = (x>mu2-1.5*sigma2) & (x<mu2+1.5*sigma2)
        x_spe = x[spe_mask]
        spe_chi2, spe_chi2_ndof = np.nan, -1
        if len(x_spe)>=5:
            bins = np.linspace(x_spe.min(),x_spe.max(),50)
            hist_counts, bin_edges = np.histogram(x_spe,bins=bins)
            bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
            bin_width = bin_edges[1]-bin_edges[0]
            expected_counts = np.clip((1-w)*norm.pdf(bin_centers,mu2,sigma2)*len(x)*bin_width,1e-6,None)
            spe_chi2 = np.sum((hist_counts-expected_counts)**2/expected_counts)
            spe_chi2_ndof = len(hist_counts)-2

        results_list.append((
            card_id,slot_id,channel_id,
            fit['mu1'],fit['sigma1'],int(np.sum(noise_mask)),
            fit['mu2'],fit['sigma2'],int(pulse_count),
            fit['gain'],fit['err_gain'],
            pulse_ratio,mu_pe,
            spe_chi2,spe_chi2_ndof
        ))

    except Exception as e:
        failed_pmts.append((pmt_label,str(e)))
        continue

# ----------------- SAVE RESULTS -----------------
dtype = np.dtype([
    ('card_id','i4'),('slot_id','i4'),('channel_id','i4'),
    ('pedestal_mean','f8'),('pedestal_sigma','f8'),('N_pedestal','i4'),
    ('spe_mean','f8'),('spe_sigma','f8'),('N_spe','i4'),
    ('gain','f8'),('gain_error','f8'),
    ('pulse_ratio','f8'),('mu_pe','f8'),
    ('spe_chi2','f8'),('spe_chi2_ndof','i4')
])
results_array = np.array(results_list,dtype=dtype)
out_dir = "/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration"
os.makedirs(out_dir,exist_ok=True)
np.savez(os.path.join(out_dir,f"NEW_doubleGauss_run2307_v3.1_chunk{chunk_id}.npz"),results=results_array)

print(f"Done. Processed PMTs {start_idx}..{end_idx-1}. Results saved.")
