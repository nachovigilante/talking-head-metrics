"""
MOD (Mouth Opening Difference) for FLAME mesh sequences.

DiffPoseTalk (Sun et al., SIGGRAPH Asia 2024) describes MOD only in prose, as
"the average difference in the size of the mouth opening between the prediction
and ground truth". This is our reconstruction (see README.md):

    open(M_t) = max_{v in S_M} y_v(t) - min_{v in S_M} y_v(t)
    MOD = (1/T) * sum_t | open(pred_t) - open(gt_t) |

with S_M the 'lips' mask of FLAME_masks.pkl (the file has no separate mouth mask)
and y the vertical axis of the FLAME template.
"""

from typing import Dict

import numpy as np
import torch

from utils.flame_utils import get_flame_model, load_flame_masks, load_mesh_sequence

VERTICAL_AXIS = 1  # FLAME's y axis


class MODCalculator:
    """Mouth Opening Difference on the 254-vertex 'lips' mask."""

    def __init__(self, device: torch.device | None = None):
        """
        Args:
            device: Device to use for computations ('cpu', 'cuda', or None for auto).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.flame_model = get_flame_model(device=self.device)
        self.mouth_opening_vertices = np.asarray(load_flame_masks()["lips"])

    def get_mouth_vertices_from_mesh(self, vertices: torch.Tensor) -> torch.Tensor:
        """Select the lip vertices: (T, 5023, 3) -> (T, 254, 3)."""
        return vertices[:, self.mouth_opening_vertices, :]

    def calculate_mouth_opening_size(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Vertical extent of the lip vertices in each frame.

        Args:
            vertices: Mesh vertices of shape (T, 5023, 3).

        Returns:
            Tensor of shape (T, 1).
        """
        y = self.get_mouth_vertices_from_mesh(vertices)[:, :, VERTICAL_AXIS]
        return y.max(dim=1, keepdim=True).values - y.min(dim=1, keepdim=True).values

    def calculate_mod(self, vertices1: torch.Tensor, vertices2: torch.Tensor) -> float:
        """
        MOD between two vertex sequences (symmetric in its arguments).

        Args:
            vertices1: First sequence (T, 5023, 3).
            vertices2: Second sequence (T, 5023, 3).

        Returns:
            Mean absolute difference of the per-frame mouth opening.
        """
        n = min(vertices1.shape[0], vertices2.shape[0])
        diff = self.calculate_mouth_opening_size(vertices1[:n]) - self.calculate_mouth_opening_size(
            vertices2[:n]
        )
        return torch.mean(torch.abs(diff.squeeze(1))).item()

    def calculate_mod_from_files(
        self, file1: str, file2: str, format1: str = "th1kh", format2: str = "th1kh"
    ) -> float:
        """MOD between two files with FLAME parameters (.npz) or vertices (.npy)."""
        vertices1 = load_mesh_sequence(file1, self.flame_model, self.device, format1)
        vertices2 = load_mesh_sequence(file2, self.flame_model, self.device, format2)
        return self.calculate_mod(vertices1, vertices2)

    def calculate_detailed_mod(
        self, vertices1: torch.Tensor, vertices2: torch.Tensor
    ) -> Dict:
        """
        MOD together with the per-frame openings and summary statistics.

        Args:
            vertices1: Predicted vertices (T, 5023, 3).
            vertices2: Ground-truth vertices (T, 5023, 3).

        Returns:
            Dict with 'mod', the per-frame openings ('mouth_opening_pred',
            'mouth_opening_gt'), their signed and absolute differences, and
            summary statistics of the absolute differences.
        """
        n = min(vertices1.shape[0], vertices2.shape[0])
        opening_pred = self.calculate_mouth_opening_size(vertices1[:n])
        opening_gt = self.calculate_mouth_opening_size(vertices2[:n])
        diff = opening_pred - opening_gt
        abs_distances = torch.abs(diff.squeeze(1))
        mod = torch.mean(abs_distances).item()
        return {
            "mod": mod,
            "mouth_opening_pred": opening_pred.cpu().numpy(),
            "mouth_opening_gt": opening_gt.cpu().numpy(),
            "mouth_opening_diff": diff.cpu().numpy(),
            "abs_distances": abs_distances.cpu().numpy(),
            "num_mouth_vertices": len(self.mouth_opening_vertices),
            "num_frames": n,
            "mean_abs_distance": mod,
            "std_abs_distance": torch.std(abs_distances).item(),
            "min_abs_distance": torch.min(abs_distances).item(),
            "max_abs_distance": torch.max(abs_distances).item(),
        }
