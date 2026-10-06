import inspect
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch

from models.flame import FLAME, FLAMEConfig
from utils.vertices_utils import load_vertices_from_file

FLAME_MASKS_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "data" / "FLAME_masks" / "FLAME_masks.pkl"
)


def _patch_for_chumpy():
    """The FLAME pickle needs chumpy, which still uses inspect.getargspec and the
    NumPy aliases (np.int, np.float, ...) removed in recent versions."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name, value in [("int", int), ("float", float), ("complex", complex),
                            ("object", object), ("unicode", str), ("str", str)]:
            if not hasattr(np, name):
                setattr(np, name, value)


def get_flame_model(config=FLAMEConfig, device=None):
    """FLAME 2020 decoder with 100 shape and 50 expression components."""
    _patch_for_chumpy()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flame = FLAME(config)
    flame.to(device)
    flame.eval()
    return flame


def load_flame_masks():
    """Vertex masks of the FLAME template (FLAME_masks.pkl, see models/data/README.md)."""
    if not FLAME_MASKS_PATH.exists():
        raise FileNotFoundError(
            f"FLAME masks not found at {FLAME_MASKS_PATH}; see models/data/README.md"
        )
    with open(FLAME_MASKS_PATH, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_mesh_sequence(path, flame_model, device, format="th1kh"):
    """Vertices (T, 5023, 3) from FLAME parameters (.npz) or from a vertex file (.npy)."""
    path = Path(path)
    if path.suffix == ".npy":
        return load_vertices_from_file(path, device)
    shape, expr, pose, _, _ = load_flame_params(str(path), device, format=format)
    with torch.no_grad():
        vertices, _, _ = flame_model(
            shape_params=shape, expression_params=expr, pose_params=pose
        )
    return vertices


def load_flame_params(fname, device, format="th1kh"):
    """
    Load a FLAME parameter sequence.

    Formats: "th1kh" (TalkingHead-1KH SMIRK reconstructions), "ensemble"
    (DiffPoseTalk inferences) and "artalk" (ARTalk exported as npz). The shape is
    set to zero (neutral identity) for every sequence, and for "th1kh" and
    "ensemble" the global rotation is centered on its mean.

    Returns:
        Tuple of (shape, expression, pose, jaw, global_rotation) tensors, where
        pose is the concatenation [global_rotation, jaw].
    """
    flame_param = np.load(fname, allow_pickle=True)
    if format == "ensemble":
        expr = flame_param["exp"].reshape(-1, 50)
        jaw = flame_param["pose"][:, 3:6].reshape(-1, 3)
        gpose = flame_param["pose"][:, 0:3].reshape(-1, 3)
        gpose = gpose - gpose.mean(axis=0, keepdims=True)
    elif format == "artalk":
        expr = flame_param["exp"].reshape((flame_param["exp"].shape[0], -1))
        gpose = flame_param["gpose"].reshape((flame_param["gpose"].shape[0], -1))
        jaw = flame_param["jaw"].reshape((flame_param["jaw"].shape[0], -1))
    elif format == "th1kh":
        expr = flame_param["expression_params"].reshape(-1, 50)
        jaw = flame_param["jaw_params"].reshape(-1, 3)
        gpose = flame_param["pose_params"][:, 0:3].reshape(-1, 3)
        gpose = gpose - gpose.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown format: {format}")

    jaw_tensor = torch.from_numpy(jaw).to(dtype=torch.float32, device=device)
    gpose_tensor = torch.from_numpy(gpose).to(dtype=torch.float32, device=device)

    full_pose_tensor = torch.cat([gpose_tensor, jaw_tensor], dim=1)
    expr_tensor = torch.from_numpy(expr).to(dtype=torch.float32, device=device)
    shape_tensor = torch.zeros(expr_tensor.shape[0], 100).to(
        dtype=torch.float32, device=device
    )

    return shape_tensor, expr_tensor, full_pose_tensor, jaw_tensor, gpose_tensor
