# Project Practice — Playbook (nprach_synch)

این فایل تمرین‌های کلیدی برگرفته از مقاله و کدهای آمادهٔ اجرا در Jupyter را به‌صورت ساختارمند ارائه می‌کند. پیشنهاد می‌شود این بخش‌ها را در Evaluate.ipynb به همان ترتیب اجرا کنید.

## 0) Prologue (در ابتدای نوت‌بوک)
```
from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings, summarize
prof = get_system_profile()
rec  = recommend_settings(prof, mode='eval')
apply_tf_settings(rec)
print(summarize(prof, rec))

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(8)
```

## 1) System setup for baseline
```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from e2e import E2E

BATCH = 16
sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
```

## 2) CFO sweep (fixed |f_off|, weighted means) — FPR/FNR vs CFO
```
import numpy as np
import matplotlib.pyplot as plt
from e2e.cfo import CFO
from parameters import CARRIER_FREQ, SAMPLING_FREQUENCY

BATCH = 32
RUNS  = 8

cfo_layer = CFO(CARRIER_FREQ, SAMPLING_FREQUENCY)
def ppm2norm(ppm):
    import tensorflow as tf
    return cfo_layer.ppm2Foffnorm(tf.constant(ppm, tf.float32)).numpy()

targets_ppm = np.array([0.0, 10.0, 20.0], dtype=float)
targets_norm = np.array([ppm2norm(x) for x in targets_ppm], dtype=float)

tol_rel = 0.1
tol_abs_norm0 = ppm2norm(0.5)

fpr_means, fnr_means, fpr_stds, fnr_stds = [], [], [], []
for t_ppm, t_norm in zip(targets_ppm, targets_norm):
    fprs_run, fnrs_run, w_fpr, w_fnr = [], [], [], []
    for _ in range(RUNS):
        snr, toa, f_off, ue_prob, fpr, fnr, toa_err, f_off_err = sys(BATCH, max_cfo_ppm=float(np.max(np.abs(targets_ppm))), ue_prob=0.5)
        f_off_np = np.abs(f_off.numpy().ravel())
        if t_ppm == 0.0:
            mask_cfo = f_off_np <= tol_abs_norm0
        else:
            mask_cfo = np.abs(f_off_np - abs(t_norm)) <= (abs(t_norm) * tol_rel)
        if not mask_cfo.any():
            continue
        fpr_np = fpr.numpy().ravel()
        fnr_np = fnr.numpy().ravel()
        fpr_valid = fpr_np[mask_cfo]
        fnr_valid = fnr_np[mask_cfo]
        fpr_valid = fpr_valid[fpr_valid >= 0.0]
        fnr_valid = fnr_valid[fnr_valid >= 0.0]
        if fpr_valid.size > 0:
            fprs_run.append(float(fpr_valid.mean()))
            w_fpr.append(int(fpr_valid.size))
        if fnr_valid.size > 0:
            fnrs_run.append(float(fnr_valid.mean()))
            w_fnr.append(int(fnr_valid.size))
    if len(fprs_run)>0:
        w = np.array(w_fpr, float); w /= w.sum()
        fpr_means.append(float(np.sum(w*np.array(fprs_run))))
        fpr_stds.append(float(np.std(fprs_run, ddof=1)) if len(fprs_run)>1 else 0.0)
    else:
        fpr_means.append(np.nan); fpr_stds.append(0.0)
    if len(fnrs_run)>0:
        w = np.array(w_fnr, float); w /= w.sum()
        fnr_means.append(float(np.sum(w*np.array(fnrs_run))))
        fnr_stds.append(float(np.std(fnrs_run, ddof=1)) if len(fnrs_run)>1 else 0.0)
    else:
        fnr_means.append(np.nan); fnr_stds.append(0.0)

plt.figure(figsize=(6,4))
plt.errorbar(targets_ppm, fpr_means, yerr=fpr_stds, fmt='o-', capsize=3, label='FPR')
plt.errorbar(targets_ppm, fnr_means, yerr=fnr_stds, fmt='s-', capsize=3, label='FNR')
plt.xlabel('CFO (ppm)')
plt.ylabel('Rate')
plt.title('Baseline: FPR/FNR vs fixed CFO (filtered, weighted)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

## 3) FNR vs SNR (binned) for CFO∈{0,10,20} ppm
```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from e2e.cfo import CFO
from parameters import CARRIER_FREQ, SAMPLING_FREQUENCY

BATCH=32
RUNS=8
NBINS=10

def bin_by_snr(snr_tensor, metric_tensor, nbins=10):
    snr = snr_tensor.numpy().ravel()
    met = metric_tensor.numpy().ravel()
    used = snr > 0.0
    if not used.any():
        return np.array([]), np.array([])
    snr_db = 10.0*np.log10(snr[used] + 1e-12)
    met_u  = met[used]
    edges = np.linspace(snr_db.min(), snr_db.max(), nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    idx = np.digitize(snr_db, edges[1:-1], right=False)
    means = []
    for b in range(nbins):
        sel = (idx == b)
        means.append(np.mean(met_u[sel]) if sel.any() else np.nan)
    return centers, np.array(means)

from e2e import E2E
sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)

cfo_layer = CFO(CARRIER_FREQ, SAMPLING_FREQUENCY)
ppm2norm = lambda ppm: cfo_layer.ppm2Foffnorm(tf.constant(ppm, tf.float32)).numpy()

targets_ppm  = [0.0, 10.0, 20.0]
targets_norm = [ppm2norm(x) for x in targets_ppm]
tol_rel = 0.1
tol_abs_norm0 = ppm2norm(0.5)

curves = []
for t_ppm, t_norm in zip(targets_ppm, targets_norm):
    xs_acc, ys_acc = [], []
    for _ in range(RUNS):
        snr, toa, f_off, ue_prob, fpr, fnr, toa_err, f_off_err = sys(BATCH, max_cfo_ppm=float(max(targets_ppm)), ue_prob=0.5)
        f_off_np = np.abs(f_off.numpy().ravel())
        if t_ppm == 0.0:
            mask = f_off_np <= tol_abs_norm0
        else:
            mask = np.abs(f_off_np - abs(t_norm)) <= (abs(t_norm)*tol_rel)
        if not mask.any():
            continue
        fnr_sel = tf.convert_to_tensor(fnr.numpy().ravel()[mask], dtype=tf.float32)
        fnr_sel = tf.boolean_mask(fnr_sel, tf.greater_equal(fnr_sel, 0.0))
        snr_sel = tf.convert_to_tensor(snr.numpy().ravel()[mask], dtype=tf.float32)
        if tf.size(fnr_sel) == 0:
            continue
        centers, means = bin_by_snr(tf.reshape(snr_sel, [-1]), tf.reshape(fnr_sel, [-1]), nbins=NBINS)
        if centers.size > 0:
            xs_acc.append(centers)
            ys_acc.append(means)
    if len(xs_acc) > 0:
        x0 = xs_acc[0]
        y_stack = np.vstack([y for y in ys_acc if y.shape == x0.shape])
        y_mean = np.nanmean(y_stack, axis=0)
        curves.append((t_ppm, x0, y_mean))

plt.figure(figsize=(6,4))
for t_ppm, x, y in curves:
    plt.plot(x, y, '-o', label=f'CFO={t_ppm:.0f} ppm')
plt.xlabel('SNR (dB)')
plt.ylabel('FNR (binned mean)')
plt.title('Baseline FNR vs SNR at fixed CFO')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

## 4) FPR/FNR vs Ptx at CFO=10 ppm (weighted)
```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from e2e import E2E
from e2e.cfo import CFO
from parameters import CARRIER_FREQ, SAMPLING_FREQUENCY

BATCH=32
RUNS=8
PTX = np.array([0.2, 0.5, 0.8], dtype=float)

sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
cfo_layer = CFO(CARRIER_FREQ, SAMPLING_FREQUENCY)
ppm2norm = lambda ppm: cfo_layer.ppm2Foffnorm(tf.constant(ppm, tf.float32)).numpy()

t_ppm=10.0
t_norm=float(ppm2norm(t_ppm))
tol_rel=0.1

fpr_m, fnr_m = [], []
for p in PTX:
    vals_fpr, vals_fnr, w_fpr, w_fnr = [], [], [], []
    for _ in range(RUNS):
        snr, toa, f_off, ue_prob, fpr, fnr, toa_err, f_off_err = sys(BATCH, max_cfo_ppm=t_ppm, ue_prob=float(p))
        f_off_np = np.abs(f_off.numpy().ravel())
        mask = np.abs(f_off_np - abs(t_norm)) <= (abs(t_norm)*tol_rel)
        if not mask.any():
            continue
        fpr_np = fpr.numpy().ravel()[mask]
        fnr_np = fnr.numpy().ravel()[mask]
        fpr_np = fpr_np[fpr_np >= 0.0]
        fnr_np = fnr_np[fnr_np >= 0.0]
        if fpr_np.size>0:
            vals_fpr.append(float(fpr_np.mean()))
            w_fpr.append(int(fpr_np.size))
        if fnr_np.size>0:
            vals_fnr.append(float(fnr_np.mean()))
            w_fnr.append(int(fnr_np.size))
    if len(vals_fpr)>0:
        w = np.array(w_fpr, float); w/=w.sum()
        fpr_m.append(float(np.sum(w*np.array(vals_fpr))))
    else:
        fpr_m.append(np.nan)
    if len(vals_fnr)>0:
        w = np.array(w_fnr, float); w/=w.sum()
        fnr_m.append(float(np.sum(w*np.array(vals_fnr))))
    else:
        fnr_m.append(np.nan)

plt.figure(figsize=(6,4))
plt.plot(PTX, fpr_m, 'o-', label='FPR')
plt.plot(PTX, fnr_m, 's-', label='FNR')
plt.xlabel('Ptx')
plt.ylabel('Rate')
plt.title('Baseline: FPR/FNR vs Ptx at CFO=10 ppm')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

## 5) Sensitivity to FFT size / #subcarriers
```
import numpy as np
import tensorflow as tf
from e2e import E2E

BATCH=16
CFG = [(24,256), (24,512), (48,256)]
for sc, fft in CFG:
    sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=sc, fft_size=fft, pfa=0.999)
    snr, toa, f_off, ue_prob, fpr, fnr, toa_err, f_off_err = sys(BATCH, max_cfo_ppm=10.0, ue_prob=0.5)
    fpr_=tf.boolean_mask(fpr, tf.greater_equal(fpr, 0.0))
    fnr_=tf.boolean_mask(fnr, tf.greater_equal(fnr, 0.0))
    print(sc, fft, float(tf.reduce_mean(fpr_).numpy()) if tf.size(fpr_)>0 else np.nan, float(tf.reduce_mean(fnr_).numpy()) if tf.size(fnr_)>0 else np.nan)
```

یادداشت‌ها
- اگر فشار حافظه داشتید، RUNS یا BATCH را کاهش دهید و nprach_num_sc=24 نگه دارید.
- برای nprach_num_sc=48، منابع WSL2 را با .wslconfig افزایش دهید و سپس دوباره اجرا کنید.
