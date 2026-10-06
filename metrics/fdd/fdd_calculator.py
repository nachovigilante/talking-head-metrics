"""
FDD (Upper-face Dynamics Deviation) for FLAME mesh sequences.

FDD, introduced in CodeTalker (Xing et al., CVPR 2023), compares how much the
upper-face vertices move over time in the ground truth and in the prediction:

    FDD = (1/|S_U|) * sum_{v in S_U} ( dyn(gt_v) - dyn(pred_v) )

where dyn(v) is the standard deviation over time of the L2 distance between the
vertex and its mean position over the sequence. The result is signed: positive
values mean the ground truth moves more than the prediction.
"""

from typing import Dict

import numpy as np
import torch

from utils.flame_utils import get_flame_model, load_flame_masks, load_mesh_sequence


class FDDCalculator:
    """FDD on the union of the 'forehead' and 'eye_region' FLAME masks (827 vertices)."""

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
        masks = load_flame_masks()
        self.upper_face_vertices = np.array(
            sorted(set(masks["forehead"]) | set(masks["eye_region"]))
        )

    def get_upper_face_vertices_from_mesh(self, vertices: torch.Tensor) -> torch.Tensor:
        """Select the upper-face vertices: (T, 5023, 3) -> (T, 827, 3)."""
        return vertices[:, self.upper_face_vertices, :]

    def calculate_dynamics(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        dyn(v) for each vertex: std over time of ||vertex(t) - mean position||_2.

        Args:
            vertices: Tensor of shape (T, V, 3).

        Returns:
            Tensor of shape (V,).
        """
        template = vertices.mean(dim=0, keepdim=True)
        distances = torch.norm(vertices - template, dim=2)
        return torch.std(distances, dim=0)

    def calculate_fdd(self, vertices1: torch.Tensor, vertices2: torch.Tensor) -> float:
        """
        Signed FDD between a prediction and the ground truth.

        Args:
            vertices1: Predicted vertices (T, 5023, 3).
            vertices2: Ground-truth vertices (T, 5023, 3).

        Returns:
            FDD; positive when the ground truth has more upper-face dynamics.
        """
        n = min(vertices1.shape[0], vertices2.shape[0])
        dyn_pred = self.calculate_dynamics(self.get_upper_face_vertices_from_mesh(vertices1[:n]))
        dyn_gt = self.calculate_dynamics(self.get_upper_face_vertices_from_mesh(vertices2[:n]))
        return torch.mean(dyn_gt - dyn_pred).item()

    def calculate_fdd_from_files(
        self,
        prediction_file: str,
        ground_truth_file: str,
        prediction_format: str = "th1kh",
        ground_truth_format: str = "th1kh",
    ) -> float:
        """FDD between two files with FLAME parameters (.npz) or vertices (.npy)."""
        pred = load_mesh_sequence(prediction_file, self.flame_model, self.device, prediction_format)
        gt = load_mesh_sequence(ground_truth_file, self.flame_model, self.device, ground_truth_format)
        return self.calculate_fdd(pred, gt)

    def calculate_detailed_fdd(
        self, vertices1: torch.Tensor, vertices2: torch.Tensor
    ) -> Dict:
        """
        Signed FDD, its unsigned variant and the per-vertex dynamics.

        Args:
            vertices1: Predicted vertices (T, 5023, 3).
            vertices2: Ground-truth vertices (T, 5023, 3).

        Returns:
            Dict with 'fdd', 'fdd_abs' (mean of |dyn_gt - dyn_pred|),
            'dynamics_pred', 'dynamics_gt', 'dynamics_diff', 'num_upper_vertices'
            and the mean and std of both dynamics.
        """
        n = min(vertices1.shape[0], vertices2.shape[0])
        dyn_pred = self.calculate_dynamics(self.get_upper_face_vertices_from_mesh(vertices1[:n]))
        dyn_gt = self.calculate_dynamics(self.get_upper_face_vertices_from_mesh(vertices2[:n]))
        dyn_diff = dyn_gt - dyn_pred
        return {
            "fdd": torch.mean(dyn_diff).item(),
            "fdd_abs": torch.mean(torch.abs(dyn_diff)).item(),
            "dynamics_pred": dyn_pred.cpu().numpy(),
            "dynamics_gt": dyn_gt.cpu().numpy(),
            "dynamics_diff": dyn_diff.cpu().numpy(),
            "num_upper_vertices": len(self.upper_face_vertices),
            "mean_dyn_pred": torch.mean(dyn_pred).item(),
            "mean_dyn_gt": torch.mean(dyn_gt).item(),
            "std_dyn_pred": torch.std(dyn_pred).item(),
            "std_dyn_gt": torch.std(dyn_gt).item(),
        }
