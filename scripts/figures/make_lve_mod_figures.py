"""Illustrations of LVE and MOD on the FLAME template (Figures 4.1 and 4.2).

The prediction is a perturbation of the ground truth (different expression and
jaw opening), so that the geometry stays realistic. Labels are in Spanish.

    python -m scripts.figures.make_lve_mod_figures [--out-dir DIR]
"""
import argparse
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from utils.flame_utils import get_flame_model, load_flame_masks
from utils.protocol_utils import FIGURES_DIR

warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

flame = get_flame_model(device=torch.device('cpu'))
LIP_IDX = np.asarray(load_flame_masks()['lips'], dtype=int)
FACES = flame.faces_tensor.cpu().numpy()
# Triangles with at least one lip vertex: the only ones where the two meshes differ.
LIP_SET = set(LIP_IDX.tolist())
LIP_FACES_MASK = np.array([any(v in LIP_SET for v in tri) for tri in FACES])


def build_mesh(jaw_open=0.16):
    """Vertices (5023, 3) of the neutral FLAME mesh with the jaw opened jaw_open radians."""
    n_shape, n_exp = 100, 50
    shape = torch.zeros(1, n_shape)
    expr  = torch.zeros(1, n_exp)
    pose  = torch.zeros(1, 6)      # global rotation (3), jaw (3)
    pose[0, 3] = jaw_open
    with torch.no_grad():
        v, _, _ = flame(shape_params=shape, expression_params=expr, pose_params=pose)
    return v[0].cpu().numpy()

def render_face(ax, verts, face_color='#f9f0e2', edge_color='#d9c7ac',
                lw=0.15, alpha=0.9, zorder=1, face_subset=None):
    """Fill the mesh triangles seen from the front (X-Y plane); only face_subset if given."""
    from matplotlib.collections import PolyCollection
    faces = FACES if face_subset is None else FACES[face_subset]
    if face_subset is None:
        z_bary = verts[faces][:, :, 2].mean(axis=1)
        front  = z_bary > np.percentile(z_bary, 30)
        faces  = faces[front]
    tri_xy = verts[faces][:, :, :2]
    coll = PolyCollection(tri_xy, facecolors=face_color, edgecolors=edge_color,
                          linewidths=lw, alpha=alpha, zorder=zorder)
    ax.add_collection(coll)


def make_lve_figure(out):
    """Ground-truth and predicted meshes overlaid, with a zoom on the pair of maximal error."""
    c_gt        = '#22c55e'
    c_gt_soft   = '#e6f5e9'
    c_gt_edge   = '#8ccfa1'
    c_pred      = '#c81e1e'
    c_pd_edge   = '#c81e1e'
    c_pd_alpha  = 0.30
    c_hi        = '#7c3aed'

    def build_mesh_expr(jaw_open=0.16, expr_vec=None):
        n_shape, n_exp = 100, 50
        shape = torch.zeros(1, n_shape)
        expr  = torch.zeros(1, n_exp) if expr_vec is None else torch.tensor(expr_vec, dtype=torch.float32).reshape(1, n_exp)
        pose  = torch.zeros(1, 6)
        pose[0, 3] = jaw_open
        with torch.no_grad():
            v, _, _ = flame(shape_params=shape, expression_params=expr, pose_params=pose)
        return v[0].cpu().numpy()

    v_gt = build_mesh_expr(jaw_open=0.16)
    expr_pred = np.zeros(50)
    expr_pred[0] = 1.5
    expr_pred[1] = -1.0
    v_pred = build_mesh_expr(jaw_open=0.16, expr_vec=expr_pred)

    lips_gt = v_gt[LIP_IDX][:, :2]
    lips_pd = v_pred[LIP_IDX][:, :2]
    d = np.linalg.norm(v_gt[LIP_IDX] - v_pred[LIP_IDX], axis=1)
    i_max = int(np.argmax(d))
    lve_mm = d[i_max] * 1000

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 4.4),
                                   gridspec_kw={'width_ratios': [1.5, 1]})

    # main panel
    render_face(ax, v_gt, face_color=c_gt_soft, edge_color=c_gt_edge,
                lw=0.18, alpha=0.85, zorder=1)
    render_face(ax, v_pred, face_color='none', edge_color=c_pd_edge,
                lw=0.25, alpha=c_pd_alpha, zorder=2)

    ax.scatter(lips_gt[:, 0], lips_gt[:, 1], s=11, c=c_gt, edgecolors='none',
               alpha=0.95, zorder=4)
    ax.scatter(lips_pd[:, 0], lips_pd[:, 1], s=11, c=c_pred, edgecolors='none',
               alpha=0.90, zorder=4)

    ax.annotate('', xy=lips_pd[i_max], xytext=lips_gt[i_max],
                arrowprops=dict(arrowstyle='->', color=c_hi, lw=2.0,
                                shrinkA=0, shrinkB=0),
                zorder=5)
    ax.scatter([lips_gt[i_max, 0]], [lips_gt[i_max, 1]], s=110, facecolors='none',
               edgecolors=c_hi, linewidths=1.7, zorder=6)

    mid_x, mid_y = (lips_gt[i_max] + lips_pd[i_max]) / 2

    x_lo = min(lips_gt[:, 0].min(), lips_pd[:, 0].min()) - 0.03
    x_hi = max(lips_gt[:, 0].max(), lips_pd[:, 0].max()) + 0.03
    y_lo = min(lips_gt[:, 1].min(), lips_pd[:, 1].min()) - 0.045
    y_hi = max(lips_gt[:, 1].max(), lips_pd[:, 1].max()) + 0.020
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    ax.annotate(r'$\max_{v \in \mathcal{S}_L}\;\|M^v - \hat{M}^v\|_2$',
                xy=(mid_x, mid_y),
                xytext=(mid_x, y_lo - 0.008),
                fontsize=10, color=c_hi, ha='center', va='top',
                arrowprops=dict(arrowstyle='-', color=c_hi, lw=0.7),
                annotation_clip=False, zorder=10)

    # right panel: zoom
    render_face(ax2, v_gt, face_color=c_gt_soft, edge_color=c_gt_edge,
                lw=0.35, alpha=0.85, zorder=1)
    render_face(ax2, v_pred, face_color='none', edge_color=c_pd_edge,
                lw=0.4, alpha=c_pd_alpha, zorder=2)

    ax2.scatter(*lips_gt[i_max], s=170, c=c_gt, alpha=0.95, zorder=5,
                edgecolors='white', linewidths=1.5)
    ax2.scatter(*lips_pd[i_max], s=170, c=c_pred, alpha=0.95, zorder=5,
                edgecolors='white', linewidths=1.5)

    ax2.annotate('', xy=lips_pd[i_max], xytext=lips_gt[i_max],
                 arrowprops=dict(arrowstyle='->', color=c_hi, lw=2.8,
                                 shrinkA=0, shrinkB=0),
                 zorder=6)

    half = 0.020
    cx, cy = (lips_gt[i_max] + lips_pd[i_max]) / 2
    ax2.set_xlim(cx - half, cx + half)
    ax2.set_ylim(cy - half, cy + half)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values(): s.set_visible(False)
    ax2.set_title(r'Zoom sobre el par de máximo error', fontsize=9.5)

    # legend below the zoom
    class ArrowHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            arrow = FancyArrowPatch((0, height/2), (width, height/2),
                                    arrowstyle='->', mutation_scale=13,
                                    color=orig_handle.get_color(),
                                    lw=orig_handle.get_linewidth())
            arrow.set_transform(trans)
            return [arrow]

    h_gt   = Line2D([0], [0], marker='o', color='none', markerfacecolor=c_gt,
                    markeredgecolor='none', markersize=8,
                    label=r'$M^v$ (ground truth)')
    h_pred = Line2D([0], [0], marker='o', color='none', markerfacecolor=c_pred,
                    markeredgecolor='none', markersize=8,
                    label=r'$\hat{M}^v$ (predicción)')
    h_arrow = Line2D([0], [0], color=c_hi, lw=2.2,
                     label=r'$d_v = \|M^v - \hat{M}^v\|_2$')

    ax2.legend(handles=[h_gt, h_pred, h_arrow],
               handler_map={h_arrow: ArrowHandler()},
               loc='upper center', bbox_to_anchor=(0.5, -0.03),
               frameon=False, fontsize=10, handletextpad=0.5,
               borderaxespad=0.5)

    # formula below the legend
    ax2.text(0.5, -0.55,
             r'$\mathrm{LVE}_t = \max_{v\in\mathcal{S}_L}\, d_v$'
             f'  ({lve_mm:.1f} mm en el ejemplo)',
             transform=ax2.transAxes, fontsize=10, ha='center', va='top', color='#333')

    fig.subplots_adjust(bottom=0.30)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def make_mod_figure(out):
    """Open and almost closed mouth, with the vertical extent open(M_t) of the lips."""
    c_gt_mod    = '#22c55e'
    c_pd_mod    = '#c81e1e'
    c_mesh_face = '#f5f5f5'
    c_mesh_edge = '#b8b8b8'
    c_mag       = '#7c3aed'   # ruler for open(M_t)

    v_open  = build_mesh(jaw_open=0.42)
    v_close = build_mesh(jaw_open=0.05)

    S_M = LIP_IDX

    def open_and_endpoints(verts, idx):
        xy = verts[idx][:, :2]
        y = xy[:, 1]
        i_top = int(np.argmax(y))
        i_bot = int(np.argmin(y))
        return y[i_top] - y[i_bot], xy[i_top], xy[i_bot]

    open_gt, top_gt, bot_gt = open_and_endpoints(v_open, S_M)
    open_pd, top_pd, bot_pd = open_and_endpoints(v_close, S_M)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), sharey=True)

    for ax, verts, o, top, bot, title, dot_color in [
        (axes[0], v_open, open_gt, top_gt, bot_gt,
         r'Ground truth: $\mathrm{open}(M_t)$', c_gt_mod),
        (axes[1], v_close, open_pd, top_pd, bot_pd,
         r'Predicción: $\mathrm{open}(\hat{M}_t)$', c_pd_mod),
    ]:
        render_face(ax, verts, face_color=c_mesh_face, edge_color=c_mesh_edge,
                    lw=0.18, alpha=0.85, zorder=1)

        ax.scatter(verts[S_M, 0], verts[S_M, 1], s=8, c=dot_color,
                   edgecolors='none', alpha=0.95, zorder=4)

        xbar = verts[S_M, 0].max() + 0.014
        ax.plot([xbar, xbar], [bot[1], top[1]], color=c_mag, lw=1.7, zorder=5)
        for y in (bot[1], top[1]):
            ax.plot([xbar - 0.003, xbar + 0.003], [y, y], color=c_mag, lw=1.5, zorder=5)
        ax.plot([top[0], xbar - 0.001], [top[1], top[1]],
                color=c_mag, lw=0.6, ls=':', zorder=4)
        ax.plot([bot[0], xbar - 0.001], [bot[1], bot[1]],
                color=c_mag, lw=0.6, ls=':', zorder=4)
        ax.text(xbar + 0.006, (top[1] + bot[1]) / 2, f'{o*1000:.1f} mm',
                fontsize=10, color=c_mag, va='center')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, fontsize=10)

    lips_open_y = v_open[S_M, 1]
    y_pad = 0.020
    y_lo = lips_open_y.min() - y_pad
    y_hi = lips_open_y.max() + y_pad
    for ax, verts in zip(axes, [v_open, v_close]):
        lips_x = verts[S_M, 0]
        x_lo = lips_x.min() - 0.03
        x_hi = lips_x.max() + 0.05
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    mod_val = abs(open_gt - open_pd) * 1000
    # formula under the panels
    fig.text(0.5, 0.10,
             r'$\mathrm{MOD}_t = \left| \mathrm{open}(M_t) - \mathrm{open}(\hat{M}_t) \right|$'
             f'  ({mod_val:.1f} mm en el ejemplo)',
             ha='center', fontsize=10, color='#333')

    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_lve_figure(args.out_dir / 'fig_lve.pdf')
    make_mod_figure(args.out_dir / 'fig_mod.pdf')
    print(f'wrote {args.out_dir}/fig_lve.pdf and fig_mod.pdf')


if __name__ == '__main__':
    main()
