"""Circle diagram for cell-cell communication network."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def net_visual_circle(
    cc_obj,
    signaling: Optional[str] = None,
    sources_use: Optional[list[str]] = None,
    targets_use: Optional[list[str]] = None,
    title: str = "",
    color_use: Optional[list[str]] = None,
    vertex_size: Optional[list[float]] = None,
    alpha: float = 0.7,
    thresh: float = 0.05,
    fig_size: tuple = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw a circle diagram of cell-cell communication.

    Parameters
    ----------
    cc_obj
        CellChat object with net computed.
    signaling
        Pathway to visualize. If None, use aggregated weight.
    sources_use
        Source cell groups to include.
    targets_use
        Target cell groups to include.
    title
        Plot title.
    color_use
        Colors for each cell group.
    vertex_size
        Vertex sizes for each cell group.
    alpha
        Transparency.
    thresh
        p-value threshold.
    fig_size
        Figure size.
    ax
        Matplotlib axes.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]
    n_groups = len(group_names)

    # Get the weight/count matrix
    if signaling and "netP" in cc and "pathways" in cc["netP"]:
        pathways = cc["netP"]["pathways"]
        if signaling in pathways:
            pw_idx = pathways.index(signaling)
            net = cc["netP"]["prob"][:, :, pw_idx]
        else:
            raise ValueError(f"Pathway '{signaling}' not found")
    elif "net" in cc and "weight" in cc["net"]:
        net = cc["net"]["weight"]
    else:
        raise RuntimeError("No network computed. Run compute_commun_prob() first.")

    # Filter by sources/targets
    src_mask = np.ones(n_groups, dtype=bool)
    tgt_mask = np.ones(n_groups, dtype=bool)
    if sources_use:
        src_mask = np.array([g in sources_use for g in group_names])
    if targets_use:
        tgt_mask = np.array([g in targets_use for g in group_names])

    # Default colors
    if color_use is None:
        cmap = plt.cm.Set2
        color_use = [cmap(i / n_groups) for i in range(n_groups)]

    # Default vertex size
    if vertex_size is None:
        # Size by total communication
        totals = net.sum(axis=1) + net.sum(axis=0)
        max_total = totals.max() if totals.max() > 0 else 1.0
        vertex_size = [0.5 + 1.5 * t / max_total for t in totals]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Place nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
    positions = {}
    for i, (angle, name) in enumerate(zip(angles, group_names)):
        x, y = np.cos(angle), np.sin(angle)
        positions[i] = (x, y)

        # Draw node
        circle = plt.Circle(
            (x, y), vertex_size[i] * 0.15,
            color=color_use[i], alpha=0.9, zorder=3
        )
        ax.add_patch(circle)

        # Label
        label_angle = np.degrees(angle)
        ha = "left" if -90 < label_angle < 90 else "right"
        ax.text(
            x * 1.3, y * 1.3, name,
            ha=ha, va="center", fontsize=10, fontweight="bold"
        )

    # Draw edges
    max_weight = net.max() if net.max() > 0 else 1.0
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            if not src_mask[i] or not tgt_mask[j]:
                continue
            weight = net[i, j]
            if weight <= 0:
                continue

            # Edge width proportional to weight
            lw = 0.5 + 4.0 * weight / max_weight

            # Draw curved arrow
            x1, y1 = positions[i]
            x2, y2 = positions[j]

            # Control point for curve
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            dx = x2 - x1
            dy = y2 - y1
            # Offset perpendicular to line
            offset = 0.2
            cx = mid_x - dy * offset
            cy = mid_y + dx * offset

            # Draw arrow
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color_use[i],
                    lw=lw,
                    alpha=alpha,
                    connectionstyle=f"arc3,rad={offset}",
                ),
                zorder=2,
            )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig
