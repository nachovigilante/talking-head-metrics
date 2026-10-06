import numpy as np
import torch

N_VERTICES = 5023


def load_vertices_from_file(npy_path, device=None):
    """
    Load a FLAME vertex sequence saved as (T, 5023, 3) or flattened (T, 15069).

    Args:
        npy_path: Path to the .npy file.
        device: Torch device (CUDA if available when None).

    Returns:
        torch.Tensor: Vertices of shape (T, 5023, 3).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices = np.load(npy_path)
    if vertices.ndim == 2 and vertices.shape[1] == N_VERTICES * 3:
        vertices = vertices.reshape(vertices.shape[0], N_VERTICES, 3)
    elif not (vertices.ndim == 3 and vertices.shape[1:] == (N_VERTICES, 3)):
        raise ValueError(
            f"Expected vertices of shape (T, {N_VERTICES}, 3) or (T, {N_VERTICES * 3}), "
            f"got {vertices.shape} in {npy_path}"
        )
    return torch.from_numpy(vertices).float().to(device)
