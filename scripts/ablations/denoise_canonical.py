"""Denoise the audio of the canonical clips with dasheng-denoiser (ablation of Section 5.3.2).

Reads DATA_DIR/TH1KH/wav and writes DATA_DIR/TH1KH/wav_denoised with the same
file names. Needs the "denoise" dependency group and dasheng-denoiser:

    uv sync --group denoise
    uv pip install git+https://github.com/xiaomi-research/dasheng-denoiser
"""
import warnings

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from dasheng_denoiser.pretrained import denoiser

from utils.protocol_utils import MANIFEST_PATH, WAV_DENOISED_DIR, WAV_DIR

warnings.filterwarnings('ignore')

WAV_DENOISED_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model = denoiser.from_ckpt(None).to(device).eval()
print("model loaded")

m = pd.read_csv(MANIFEST_PATH)
print(f"manifest: {len(m)} clips")

ok, skip, fail = 0, 0, 0
for w in tqdm(m.wav, total=len(m), desc="denoise"):
    src_p = WAV_DIR / w
    dst_p = WAV_DENOISED_DIR / w
    if dst_p.exists():
        skip += 1; continue
    try:
        y, sr = torchaudio.load(str(src_p))
        if sr != 16000:
            y = torchaudio.functional.resample(y, sr, 16000); sr = 16000
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
        y = y.to(device)
        with torch.no_grad():
            y_hat = model(y).detach().cpu()
        torchaudio.save(str(dst_p), y_hat, sr)
        ok += 1
    except Exception as e:
        fail += 1
        print(f"  fail {w[:40]}: {e}")

print(f"\ndone: {ok} denoised, {skip} already present, {fail} failed")
