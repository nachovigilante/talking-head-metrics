"""Prosodic events of a clip with its word-aligned transcription (Figures 4.3, 4.4 and 5.2).

    python -m scripts.figures.make_prosodic_events_figure [--denoised] CLIP [CLIP ...]
        fig_eventos_prosodicos_<i>.pdf, one per clip: the four prosodic signals,
        the detected peaks and the words. The silence before the first peak and
        after the last one is left out of the plot. The thesis figures use
        --denoised (audio after scripts/ablations/denoise_canonical.py) so that
        the signals read better; the evaluation itself uses the raw audio.
    python -m scripts.figures.make_prosodic_events_figure --compare CLIP
        fig_eventos_crudo_vs_filtrado.pdf: F0 and onset strength on raw and
        denoised audio, side by side.
    python -m scripts.figures.make_prosodic_events_figure [--denoised] --video CLIP
        The same figure with a cursor synchronized with the audio, at normal and
        half speed (needs ffmpeg).

The figures are written to results/figures/. Their inputs are in inputs/ next to
this script: words_<clip>.json, the word timings from faster-whisper (cached so
that the figures can be rebuilt without running Whisper), and
listening_<clip>.csv, the author's judgment of each peak after listening to the
clip ("emphasis" or "doubtful"); doubtful peaks are drawn hollow. Labels are in
Spanish, as in the thesis.
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from metrics.pbas.prosodic_extractor import ProsodicExtractor
from utils.protocol_utils import FIGURES_DIR, WAV_DENOISED_DIR, WAV_DIR

INPUTS_DIR = Path(__file__).resolve().parent / "inputs"
FPS = 25

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_SIGNAL = '#1f4e79'
C_PEAK = '#d1495b'
C_BAND = '#e9ecef'  # alternate words


def judgments_for(stem):
    """(signal, time, judgment) for each peak the author listened to in this clip."""
    import csv
    f = INPUTS_DIR / f'listening_{stem}.csv'
    if not f.exists():
        return []
    with open(f, newline='') as fh:
        return [(r['signal'], float(r['time_s']), r['judgment']) for r in csv.DictReader(fh)]


def at(peaks, sig):
    """Times (s) and signal values at peaks given in (possibly fractional) frames."""
    peaks = np.asarray(peaks, dtype=float)
    return peaks / FPS, np.interp(peaks, np.arange(len(sig)), sig)


def words_for(stem, wav):
    cache = INPUTS_DIR / f'words_{stem}.json'  # transcribed from the raw audio
    if cache.exists():
        return json.loads(cache.read_text())
    from faster_whisper import WhisperModel
    model = WhisperModel('small.en', device='cpu', compute_type='int8')
    segs, _ = model.transcribe(str(wav), language='en', beam_size=5, word_timestamps=True)
    words = [[w.word.strip(), round(w.start, 3), round(w.end, 3)] for s in segs for w in (s.words or [])]
    cache.write_text(json.dumps(words, ensure_ascii=False, indent=0))
    return words


def f0_native(wav, ex):
    """F0 at Praat's own times, before resampling: (times, F0 with NaN when unvoiced, interpolated F0)."""
    import parselmouth
    pitch = parselmouth.Sound(str(wav)).to_pitch_ac(
        time_step=1.0 / FPS, pitch_floor=ex.f0_floor, pitch_ceiling=ex.f0_ceiling)
    tp, f0 = pitch.xs(), pitch.selected_array['frequency'].copy()
    f0[f0 == 0] = np.nan
    voiced = ~np.isnan(f0)
    f0i = np.interp(tp, tp[voiced], f0[voiced]) if voiced.sum() >= 3 else np.zeros_like(tp)
    return tp, f0, f0i


def build_figure(stem, ex, wav_dir):
    wav = wav_dir / f'{stem}.wav'
    ev = ex.extract_all(wav)
    words = words_for(stem, wav)
    n = min(len(ev[k]['signal']) for k in ev)
    t = np.arange(n) / FPS
    x0, x1 = x_range(ev, n)
    judgments = judgments_for(stem)
    used = set()
    def split_peaks(key, peaks):
        """Split off the peaks judged doubtful; a judgment goes to the closest peak within 60 ms."""
        firm, doubtful = [], []
        for p in peaks:
            cand = [(abs(p / FPS - tj), i) for i, (sj, tj, _) in enumerate(judgments)
                    if sj == key and abs(p / FPS - tj) <= 0.06]
            if not cand:
                firm.append(p)
                if judgments:
                    print(f'  no judgment for {key} peak at {p / FPS:.2f} s')
                continue
            i = min(cand)[1]; used.add(i)
            (doubtful if judgments[i][2] == 'doubtful' else firm).append(p)
        return np.array(firm, dtype=float), np.array(doubtful, dtype=float)
    any_doubtful = any(j == 'doubtful' for _, _, j in judgments)
    panels = [
        ('pitch_accent', 'F0 (Hz)'),
        ('onset', 'Intensidad\nde inicio'),
        ('energy', 'Energía RMS'),
        ('prominence', 'Prominencia'),
    ]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(6.0, 4.9), sharex=True,
                             gridspec_kw={'height_ratios': [0.85, 1, 1, 1, 1], 'hspace': 0.14})
    ax = axes[0]
    for i, (w, a, b) in enumerate(words):
        if i % 2 == 0:
            ax.axvspan(a, b, color=C_BAND, lw=0)
        ax.text((max(a, x0) + min(b, x1)) / 2, (0.82, 0.5, 0.18)[i % 3], w, ha='center', va='center', fontsize=7)
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_ylabel('Palabras', rotation=0, ha='right', va='center', labelpad=6)
    for ax, (key, label) in zip(axes[1:], panels):
        sig = ev[key]['signal'][:n]; peaks = ev[key]['beats']; peaks = peaks[peaks < n]
        for i, (w, a, b) in enumerate(words):
            if i % 2 == 0:
                ax.axvspan(a, b, color=C_BAND, lw=0, zorder=0)
        firm, doubtful = split_peaks(key, peaks)
        def pos(pk, sig=sig):
            return at(pk, sig)
        if key == 'pitch_accent':
            tp, raw, f0i = f0_native(wav, ex)
            ax.plot(tp, f0i, color=C_SIGNAL, lw=0.8, ls=':', zorder=2, label='F0 interpolada')
            ax.plot(tp, raw, color=C_SIGNAL, lw=1.2, zorder=3, label='F0 (tramos sonoros)')
            def pos(pk, tp=tp, f0i=f0i):  # on the F0 contour drawn above, at Praat's times
                times = np.asarray(pk, float) / FPS
                return times, np.interp(times, tp, f0i)
            ax.plot(*pos(firm), marker='v', ls='none', ms=4.5, color=C_PEAK, zorder=4, label='pico detectado')
            if any_doubtful:
                ax.plot([], [], marker='v', ls='none', ms=4.5, markerfacecolor='white', markeredgecolor=C_PEAK, label='pico dudoso al oído')
            fig.legend(loc='upper center', bbox_to_anchor=(0.55, 1.0), frameon=False, fontsize=7, ncol=4, handlelength=1.8)
        else:
            ax.plot(t, sig, color=C_SIGNAL, lw=1.0, zorder=3)
            ax.plot(*pos(firm), marker='v', ls='none', ms=4.5, color=C_PEAK, zorder=4)
        if len(doubtful):
            ax.plot(*pos(doubtful), marker='v', ls='none', ms=4.5, markerfacecolor='white', markeredgecolor=C_PEAK, zorder=5)
        ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=6)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.grid(axis='y', color='#dddddd', lw=0.4)
    for i, (sj, tj, _) in enumerate(judgments):
        if i not in used:
            print(f'  judgment without a peak: {sj} at {tj:.2f} s')
    axes[-1].set_xlabel('Tiempo (s)')
    axes[-1].set_xlim(x0, x1)
    return fig, axes, t, ev, words, wav


def x_range(ev, n, margin=0.3):
    """Limits of the x axis: the stretches before the first peak and after the last one
    are cut, keeping a margin and rounding outward to the half second."""
    beats = np.concatenate([ev[k]['beats'][ev[k]['beats'] < n] for k in ev]) / FPS
    start = max(0.0, np.floor((beats.min() - margin) * 2) / 2)
    end = min((n - 1) / FPS, np.ceil((beats.max() + margin) * 2) / 2)
    return start, end


def draw(stem, ex, wav_dir, out_pdf, out_png=None):
    fig, axes, t, ev, words, _ = build_figure(stem, ex, wav_dir)
    fig.savefig(out_pdf, bbox_inches='tight')
    if out_png:
        fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return {k: int(len(ev[k]['beats'])) for k in ev}, words


def video(stem, ex, wav_dir, out_dir, suffix):
    """The figure with a cursor advancing at 25 fps, muxed with the audio."""
    import subprocess
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes, t, ev, words, wav = build_figure(stem, ex, wav_dir)
    fig.set_dpi(220)
    fig.subplots_adjust(left=0.19, right=0.975, top=0.90, bottom=0.09)
    cursors = [ax.axvline(0, color='#222222', lw=1.1, zorder=6) for ax in axes]
    span = axes[0].axvspan(0, 0, color='#ffd166', alpha=0.55, lw=0, zorder=1)
    clock = axes[0].text(0.005, 1.25, '', transform=axes[0].transAxes, fontsize=7, ha='left', va='bottom')
    keys = ['pitch_accent', 'onset', 'energy', 'prominence']
    n = len(t)
    active = []
    for ax, key in zip(axes[1:], keys):
        sc = ax.scatter([], [], s=70, marker='v', facecolor='none', edgecolor='#222222', lw=1.2, zorder=7)
        active.append((sc, ev[key]['beats'], ev[key]['signal'][:n]))

    def frame(i):
        ti = t[i]
        for c in cursors:
            c.set_xdata([ti, ti])
        cur = [(a, b) for w, a, b in words if a <= ti < b]
        if cur:
            a, b = cur[0]
            span.set_x(a); span.set_width(b - a)
        clock.set_text(f't = {ti:5.2f} s')
        for sc, beats, sig in active:
            near = [p for p in beats if abs(p - i) <= 1 and p < n]
            sc.set_offsets(np.c_[at(near, sig)] if near else np.empty((0, 2)))
        return cursors

    silent = out_dir / f'{stem}{suffix}_silent.mp4'
    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    anim.save(str(silent), writer=FFMpegWriter(fps=FPS, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']))
    plt.close(fig)
    out1 = out_dir / f'{stem}{suffix}_1x.mp4'
    out05 = out_dir / f'{stem}{suffix}_0.5x.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(silent), '-i', str(wav),
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(out1)], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(silent), '-i', str(wav),
                    '-filter_complex', '[0:v]setpts=2.0*PTS[v];[1:a]atempo=0.5[a]',
                    '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-shortest', str(out05)], check=True)
    silent.unlink()
    return out1, out05


def compare(stem, ex, out_pdf, out_png=None):
    """F0 and onset strength of a clip on raw and denoised audio."""
    words = words_for(stem, WAV_DIR / f'{stem}.wav')
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.4), sharex=True,
                             gridspec_kw={'hspace': 0.18, 'wspace': 0.18})
    for col, (label, wav_dir) in enumerate([('audio crudo', WAV_DIR), ('audio filtrado', WAV_DENOISED_DIR)]):
        ev = ex.extract_all(wav_dir / f'{stem}.wav')
        n = min(len(ev[k]['signal']) for k in ev); t = np.arange(n) / FPS
        for row, key in enumerate(['pitch_accent', 'onset']):
            ax = axes[row, col]; sig = ev[key]['signal'][:n]; pk = ev[key]['beats']; pk = pk[pk < n]
            for i, (w, a, b) in enumerate(words):
                if i % 2 == 0:
                    ax.axvspan(a, b, color=C_BAND, lw=0, zorder=0)
            if key == 'pitch_accent':
                tp, raw, f0i = f0_native(wav_dir / f'{stem}.wav', ex)
                ax.plot(tp, f0i, color=C_SIGNAL, lw=0.7, ls=':', zorder=2)
                ax.plot(tp, raw, color=C_SIGNAL, lw=1.1, zorder=3)
                ax.set_ylim(60, 650)
            else:
                ax.plot(t, sig, color=C_SIGNAL, lw=0.9, zorder=3)
            if key == 'pitch_accent':
                ax.plot(pk / FPS, np.interp(pk / FPS, tp, f0i), marker='v', ls='none', ms=4, color=C_PEAK, zorder=4)
            else:
                ax.plot(*at(pk, sig), marker='v', ls='none', ms=4, color=C_PEAK, zorder=4)
            ax.grid(axis='y', color='#dddddd', lw=0.4)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            if row == 0:
                ax.set_title(f"{label}: {len(ev['pitch_accent']['beats'])} acentos, {len(ev['onset']['beats'])} inicios", fontsize=8)
            if col == 0:
                ax.set_ylabel(['F0 (Hz)', 'Intensidad\nde inicio'][row], rotation=0, ha='right', va='center', labelpad=6)
        axes[1, col].set_xlabel('Tiempo (s)'); axes[1, col].set_xlim(0, t[-1])
    fig.savefig(out_pdf, bbox_inches='tight')
    if out_png:
        fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('clips', nargs='+', help='clip stems (data/manifest_canonical.csv)')
    parser.add_argument('--denoised', action='store_true', help='use the denoised audio')
    parser.add_argument('--compare', action='store_true', help='raw vs denoised comparison')
    parser.add_argument('--video', action='store_true', help='animated version with audio')
    parser.add_argument('--out-dir', type=Path, default=FIGURES_DIR)
    parser.add_argument('--png', action='store_true', help='also save a PNG preview')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ex = ProsodicExtractor()

    if args.compare:
        for stem in args.clips:
            out = args.out_dir / 'fig_eventos_crudo_vs_filtrado.pdf'
            compare(stem, ex, out, out.with_suffix('.png') if args.png else None)
            print('wrote', out)
        return
    wav_dir, suffix = (WAV_DENOISED_DIR, '_denoised') if args.denoised else (WAV_DIR, '')
    if args.video:
        for stem in args.clips:
            for out in video(stem, ex, wav_dir, args.out_dir / 'videos', suffix):
                print('wrote', out)
        return
    for i, stem in enumerate(args.clips, 1):
        out = args.out_dir / f'fig_eventos_prosodicos_{i}.pdf'
        counts, words = draw(stem, ex, wav_dir, out, out.with_suffix('.png') if args.png else None)
        print(stem, counts, ' '.join(w for w, _, _ in words))


if __name__ == '__main__':
    main()
