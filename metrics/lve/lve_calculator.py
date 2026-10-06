"""
LVE (Lip Vertex Error) for FLAME mesh sequences.

LVE, introduced in MeshTalk (Richard et al., ICCV 2021), is the maximal L2 error
over the lip vertices in each frame, averaged over all frames.
"""

import numpy as np
import torch

from utils.flame_utils import get_flame_model, load_flame_masks, load_mesh_sequence


class LVECalculator:
    """Lip Vertex Error between two FLAME mesh sequences, on the 254-vertex 'lips' mask."""

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
        self.lip_vertices = np.asarray(load_flame_masks()["lips"])

    def get_lip_vertices_from_mesh(self, vertices: torch.Tensor) -> torch.Tensor:
        """Select the lip vertices: (T, 5023, 3) -> (T, 254, 3)."""
        return vertices[:, self.lip_vertices, :]

    def calculate_frame_lve(
        self, vertices1: torch.Tensor, vertices2: torch.Tensor
    ) -> torch.Tensor:
        """
        Maximal lip vertex error of each frame.

        Args:
            vertices1: First sequence vertices (T, 5023, 3).
            vertices2: Second sequence vertices (T, 5023, 3).

        Returns:
            Tensor of shape (T,).
        """
        l2_distances = torch.norm(
            self.get_lip_vertices_from_mesh(vertices1)
            - self.get_lip_vertices_from_mesh(vertices2),
            dim=2,
        )
        return torch.max(l2_distances, dim=1).values

    def calculate_sequence_lve(
        self, vertices1: torch.Tensor, vertices2: torch.Tensor
    ) -> float:
        """LVE of a sequence: the per-frame maximum averaged over frames."""
        return torch.mean(self.calculate_frame_lve(vertices1, vertices2)).item()

    def calculate_lve_from_files(
        self, file1: str, file2: str, format1: str = "th1kh", format2: str = "th1kh"
    ) -> float:
        """
        LVE between two files with FLAME parameters (.npz) or vertices (.npy).

        Both sequences are truncated to the shorter length.
        """
        vertices1 = load_mesh_sequence(file1, self.flame_model, self.device, format1)
        vertices2 = load_mesh_sequence(file2, self.flame_model, self.device, format2)
        n = min(vertices1.shape[0], vertices2.shape[0])
        return self.calculate_sequence_lve(vertices1[:n], vertices2[:n])

    def calculate_detailed_lve(
        self, vertices1: torch.Tensor, vertices2: torch.Tensor
    ) -> dict:
        """
        LVE together with per-frame values and summary statistics.

        Args:
            vertices1: First sequence vertices (T, 5023, 3).
            vertices2: Second sequence vertices (T, 5023, 3).

        Returns:
            Dict with 'mean_lve' (the LVE), 'std_lve', 'min_lve', 'max_lve',
            'frame_lves', 'mean_lip_vertex_error', 'max_lip_vertex_error',
            'num_frames' and 'num_lip_vertices'.
        """
        frame_lves = self.calculate_frame_lve(vertices1, vertices2)
        all_l2_distances = torch.norm(
            self.get_lip_vertices_from_mesh(vertices1)
            - self.get_lip_vertices_from_mesh(vertices2),
            dim=2,
        )
        return {
            "mean_lve": torch.mean(frame_lves).item(),
            "std_lve": torch.std(frame_lves).item(),
            "min_lve": torch.min(frame_lves).item(),
            "max_lve": torch.max(frame_lves).item(),
            "frame_lves": frame_lves.cpu().numpy(),
            "mean_lip_vertex_error": torch.mean(all_l2_distances).item(),
            "max_lip_vertex_error": torch.max(all_l2_distances).item(),
            "num_frames": vertices1.shape[0],
            "num_lip_vertices": len(self.lip_vertices),
        }
