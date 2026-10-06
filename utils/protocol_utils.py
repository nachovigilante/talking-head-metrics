"""Pieces of the evaluation protocol shared by the scripts that run on the canonical set.

data/manifest_canonical.csv and data/language_ids_final.csv are part of the
repository. The rest of the data is not; it goes under DATA_DIR, which defaults
to <repo>/data and can be set with the DATA_DIR environment variable:

    TH1KH/wav/            speech of each clip, 16 kHz
    TH1KH/wav_denoised/   the same audio after dasheng-denoiser
                          (scripts/ablations/denoise_canonical.py)
    TH1KH/npz/            SMIRK reconstructions of the TalkingHead-1KH clips (ground truth)
    DiffPoseTalk/         DiffPoseTalk inferences (.npz)
    ARTalk/               ARTalk inferences ((T, 106) tensors saved with torch.save)
    MultiTalk/            MultiTalk inferences (vertices, (T, 15069) .npy)
    CodeTalker/           CodeTalker inferences (vertices, (T, 15069) .npy)

Decoding ARTalk's motion needs ARTalk's own FLAME implementation: set ARTALK_REPO
to a clone of https://github.com/xg-chu/ARTalk.
"""
import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

from metrics.bas.bas_calculator import beat_alignment_score
from metrics.pbas.pbas_calculator import MOTION_TYPES, PROSODIC_TYPES
from utils.flame_utils import load_flame_params
from utils.vertices_utils import load_vertices_from_file

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest_canonical.csv"
LANGUAGE_IDS_PATH = REPO_ROOT / "data" / "language_ids_final.csv"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
WAV_DIR = DATA_DIR / "TH1KH" / "wav"
WAV_DENOISED_DIR = DATA_DIR / "TH1KH" / "wav_denoised"
GT_NPZ_DIR = DATA_DIR / "TH1KH" / "npz"
DPT_DIR = DATA_DIR / "DiffPoseTalk"
ARTALK_DIR = DATA_DIR / "ARTalk"
MULTITALK_DIR = DATA_DIR / "MultiTalk"
CODETALKER_DIR = DATA_DIR / "CodeTalker"

SHUFFLE_SEED = 42
PBAS_KEYS = [f"pbas_{bt}_{pt}" for bt in MOTION_TYPES for pt in PROSODIC_TYPES]


def derangement(video_ids, seed=SHUFFLE_SEED):
    """Pair each video with another one (no fixed points), deterministically.

    Every script uses the same seed, so the shuffled pairs are the same throughout.
    """
    n = len(video_ids)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    for i in range(n):
        if indices[i] == i:
            j = (i + 1) % n
            indices[i], indices[j] = indices[j], indices[i]
    assert all(indices[i] != i for i in range(n)), "derangement failed"
    return {video_ids[i]: video_ids[indices[i]] for i in range(n)}


# ---- Head rotation --------------------------------------------------------------

def load_rot_gt(path, device):
    """Head rotation (T, 3), axis-angle, from a TalkingHead-1KH SMIRK npz."""
    _, _, pose, _, _ = load_flame_params(str(path), device, format="th1kh")
    return pose[:, :3].cpu().numpy()


def load_rot_dpt(path, device):
    """Head rotation (T, 3), axis-angle, from a DiffPoseTalk inference npz."""
    _, _, pose, _, _ = load_flame_params(str(path), device, format="ensemble")
    return pose[:, :3].cpu().numpy()


def load_rot_artalk(path, device):
    """Head rotation (T, 3), axis-angle, from an ARTalk inference (T, 106) tensor.

    Unlike load_rot_gt and load_rot_dpt, the rotation is not centered on its mean.
    ARTalk's mean head rotation is close to zero (0.7 degrees on average over a
    sample of 200 canonical clips, against about 16 for the ground truth), so
    centering it would change the rotations very little.
    """
    motion = torch.load(path, map_location=device, weights_only=True).float()
    return np.asarray(motion[..., 100:103].cpu().numpy())


# ---- Vertices -------------------------------------------------------------------

def load_artalk_flame(device, artalk_repo=None):
    """ARTalk's FLAME decoder (300 shape, 100 expression components), loaded from its repo."""
    artalk_repo = Path(artalk_repo or os.environ.get("ARTALK_REPO", ""))
    flame_dir = artalk_repo / "app" / "flame_model"
    if not (flame_dir / "FLAME.py").exists():
        raise FileNotFoundError(
            "ARTalk's FLAME model not found; set ARTALK_REPO to a clone of "
            "https://github.com/xg-chu/ARTalk"
        )
    pkg_name = "_artalk_flame_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(flame_dir)]
        sys.modules[pkg_name] = pkg
        for sub in ("lbs", "FLAME"):
            spec = importlib.util.spec_from_file_location(f"{pkg_name}.{sub}", flame_dir / f"{sub}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkg_name}.{sub}"] = mod
            spec.loader.exec_module(mod)
    flame_model = sys.modules[f"{pkg_name}.FLAME"].FLAMEModel
    return flame_model(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).to(device).eval()


@torch.no_grad()
def decode_flame_npz(flame, path, device, format, canonical=False):
    """Vertices of a FLAME parameter file; with canonical=True the global rotation is zeroed."""
    shape, expr, pose, _, _ = load_flame_params(str(path), device, format=format)
    if canonical:
        pose = pose.clone()
        pose[:, :3] = 0.0
    vertices, _, _ = flame(shape_params=shape, expression_params=expr, pose_params=pose)
    return vertices


@torch.no_grad()
def decode_artalk(flame_artalk, path, device, canonical=False):
    """Vertices of an ARTalk inference: 100 expression and 6 pose values (global, jaw) per frame.

    As in load_rot_artalk, the global rotation is used as predicted, not centered
    on its mean (see that function).
    """
    motion = torch.load(path, map_location=device, weights_only=True).float()
    expression = motion[..., :100]
    pose = motion[..., 100:].clone()
    if canonical:
        pose[:, :3] = 0.0
    shape = motion.new_zeros(motion.shape[0], 300)
    return flame_artalk(shape_params=shape, expression_params=expression, pose_params=pose)


def load_vertices(path, device):
    """Vertices of MultiTalk or CodeTalker, which predict meshes in canonical orientation."""
    return load_vertices_from_file(str(path), device)


# ---- Beat scores ----------------------------------------------------------------

def bas_scores(motion_beats, ref_beats, sigma=3.0):
    """BAS of each motion beat type against the same beat type of a reference motion."""
    return {
        f"bas_{bt}": beat_alignment_score(motion_beats[bt]["beats"], ref_beats[bt]["beats"], sigma)
        for bt in MOTION_TYPES
    }


def pbas_scores(motion_beats, prosodic, sigma=3.0):
    """PBAS of the 12 motion x prosodic combinations (NaN if the prosody is missing)."""
    if prosodic is None:
        return {k: float("nan") for k in PBAS_KEYS}
    return {
        f"pbas_{bt}_{pt}": beat_alignment_score(
            motion_beats[bt]["beats"], prosodic[pt]["beats"], sigma
        )
        for bt in MOTION_TYPES
        for pt in PROSODIC_TYPES
    }
