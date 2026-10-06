"""Cross-correlation between head motion and speech (Figure 5.1).

For 200 canonical clips, normalized cross-correlation between ||omega(t)|| of
the ground-truth head motion and the onset strength (and, separately, the RMS
energy) of the speech, for lags of up to 1 s. The figure shows the median and
interquartile range over clips, for true pairs (motion and its own audio) and
shuffled pairs (motion and another clip's audio). Labels are in Spanish.

    python -m scripts.figures.make_xcorr_figure [--out-dir DIR]
"""
import argparse
import warnings

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import torch

from metrics.bas.so3_beat_detector import SO3BeatDetector
from metrics.pbas.prosodic_extractor import ProsodicExtractor
from utils.flame_utils import load_flame_params
from utils.protocol_utils import FIGURES_DIR, GT_NPZ_DIR, MANIFEST_PATH, WAV_DIR, derangement

warnings.filterwarnings('ignore')

N_SAMPLE = 200
MAX_LAG_SEC = 1.0
FPS = 25
MAX_LAG = int(MAX_LAG_SEC * FPS)
SEED = 42

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})


def zscore(x):
    x = np.asarray(x, float)
    x[np.isnan(x)] = 0
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def xcorr_norm(a, b, max_lag):
    a = zscore(a); b = zscore(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L < 0:
            x = a[-L:]; y = b[:n + L]
        elif L > 0:
            x = a[:n - L]; y = b[L:]
        else:
            x = a; y = b
        m = min(len(x), len(y))
        if m < 5:
            out[i] = np.nan; continue
        out[i] = float(np.mean(x[:m] * y[:m]))
    return lags / FPS, out


def omega_magnitude(rot, detector=SO3BeatDetector()):
    """||omega(t)|| on the video frame grid, as used by BAS and PBAS."""
    return detector.detect_all_beats_so3(rot)["omega"]["signal"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH)
    sample = manifest.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    extractor = ProsodicExtractor()
    n = len(sample)
    idxs = derangement(list(range(n)), SEED)

    omega_cache, onset_cache, rms_cache = [], [], []
    for i, r in tqdm(list(sample.iterrows()), total=n, desc='cache'):
        try:
            _, _, pose, _, _ = load_flame_params(str(GT_NPZ_DIR / r.gt_npz),
                                                  torch.device('cpu'), format='th1kh')
            rot = pose[:, :3].cpu().numpy()
            om = omega_magnitude(rot)
            feat = extractor.extract_all(str(WAV_DIR / r.wav))
            on = feat['onset']['signal']
            en = feat['energy']['signal']
        except Exception as e:
            print(f'skip {r.clip_stem[:30]}: {e}')
            omega_cache.append(None); onset_cache.append(None); rms_cache.append(None)
            continue
        omega_cache.append(om); onset_cache.append(on); rms_cache.append(en)

    xc_true_on, xc_shuf_on, xc_true_rms, xc_shuf_rms = [], [], [], []
    lags_axis = None
    for i in range(n):
        if omega_cache[i] is None: continue
        if onset_cache[i] is not None:
            lags, c = xcorr_norm(omega_cache[i], onset_cache[i], MAX_LAG)
            xc_true_on.append(c); lags_axis = lags
        if rms_cache[i] is not None:
            _, c = xcorr_norm(omega_cache[i], rms_cache[i], MAX_LAG)
            xc_true_rms.append(c)
        j = idxs[i]
        if onset_cache[j] is not None:
            _, c = xcorr_norm(omega_cache[i], onset_cache[j], MAX_LAG)
            xc_shuf_on.append(c)
        if rms_cache[j] is not None:
            _, c = xcorr_norm(omega_cache[i], rms_cache[j], MAX_LAG)
            xc_shuf_rms.append(c)

    def agg(arrs):
        a = np.array(arrs)
        return np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0)

    med_t_on, lo_t_on, hi_t_on = agg(xc_true_on)
    med_s_on, lo_s_on, hi_s_on = agg(xc_shuf_on)
    med_t_rms, lo_t_rms, hi_t_rms = agg(xc_true_rms)
    med_s_rms, lo_s_rms, hi_s_rms = agg(xc_shuf_rms)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    for ax, (title, mt, lt, ht, ms, ls, hs) in zip(axes, [
        (r'$\|\omega(t)\|$ vs. intensidad de inicio',
         med_t_on, lo_t_on, hi_t_on, med_s_on, lo_s_on, hi_s_on),
        (r'$\|\omega(t)\|$ vs. energía RMS',
         med_t_rms, lo_t_rms, hi_t_rms, med_s_rms, lo_s_rms, hi_s_rms),
    ]):
        ax.axvline(0, color='0.55', lw=0.6, ls='--', zorder=1)
        ax.axhline(0, color='0.55', lw=0.6, ls='--', zorder=1)
        ax.fill_between(lags_axis, ls, hs, color='#bbb', alpha=0.5, lw=0, zorder=2)
        ax.plot(lags_axis, ms, color='#555', lw=1.1, label='pares permutados', zorder=3)
        ax.fill_between(lags_axis, lt, ht, color='#7ec5e0', alpha=0.55, lw=0, zorder=4)
        ax.plot(lags_axis, mt, color='#1f6f9f', lw=1.6, label='pares verdaderos', zorder=5)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(lags_axis.min(), lags_axis.max())
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.legend(loc='upper right', frameon=False, fontsize=8)
    axes[0].set_ylabel('correlación cruzada normalizada', fontsize=9)

    fig.supxlabel(r'desfase $\tau$ (s) entre $\|\omega(t)\|$ y la señal acústica'
                  '\n'
                  r'($\tau>0$: el movimiento precede al audio;  $\tau<0$: el audio precede al movimiento)',
                  fontsize=9, y=-0.02)

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / 'xcorr_motion_audio.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')

    print('\nmedian correlation at lag 0:')
    print(f'  onset true: {med_t_on[MAX_LAG]:.3f}  shuf: {med_s_on[MAX_LAG]:.3f}')
    print(f'  rms   true: {med_t_rms[MAX_LAG]:.3f}  shuf: {med_s_rms[MAX_LAG]:.3f}')


if __name__ == '__main__':
    main()
