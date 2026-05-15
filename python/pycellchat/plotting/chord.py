"""Chord diagram for cell-cell communication."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path


def net_visual_chord_cell(
    cc_obj,
    signaling: Optional[str] = None,
    sources_use: Optional[list[str]] = None,
    targets_use: Optional[list[str]] = None,
    title: str = "",
    color_use: Optional[list[str]] = None,
    transparency: float = 0.5,
    thresh: float = 0.05,
    reduce: float = 0.9,
    fig_size: tuple = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw a chord diagram of cell-cell communication.

    Parameters
    ----------
    cc_obj
        CellChat object with net computed.
    signaling
        Pathway to show. If None, use aggregated weight.
    sources_use
        Source groups to include.
    targets_use
        Target groups to include.
    title
        Plot title.
    color_use
        Colors for each group.
    transparency
        Chord transparency.
    thresh
        p-value threshold.
    reduce
        Gap between arcs (0-1).
    fig_size
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]
    n_groups = len(group_names)

    # Get weight matrix
    if signaling and "netP" in cc:
        pathways = cc["netP"].get("pathways", [])
        if signaling in pathways:
            pw_idx = pathways.index(signaling)
            net = cc["netP"]["prob"][:, :, pw_idx]
        else:
            raise ValueError(f"Pathway '{signaling}' not found")
    elif "net" in cc and "weight" in cc["net"]:
        net = cc["net"]["weight"]
    else:
        raise RuntimeError("No network computed")

    if color_use is None:
        cmap = plt.cm.Set2
        color_use = [cmap(i / n_groups) for i in range(n_groups)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Total flow per group (in + out)
    row_sums = net.sum(axis=1)
    col_sums = net.sum(axis=0)
    totals = row_sums + col_sums
    total_sum = totals.sum()

    if total_sum == 0:
        ax.text(0, 0, "No communication", ha="center", va="center")
        return fig

    # Arc angles for each group
    gap_fraction = reduce * 0.05
    available_angle = 2 * np.pi - n_groups * gap_fraction * 2 * np.pi
    angles = []
    start = 0
    for i in range(n_groups):
        frac = totals[i] / total_sum
        arc = frac * available_angle
        angles.append((start, start + arc))
        start += arc + gap_fraction * 2 * np.pi

    # Draw arcs
    for i in range(n_groups):
        theta1, theta2 = angles[i]
        if theta2 <= theta1:
            continue

        # Outer arc
        arc = mpatches.Wedge(
            (0, 0), 1.0, np.degrees(theta1), np.degrees(theta2),
            width=0.1, facecolor=color_use[i], edgecolor="white", linewidth=0.5
        )
        ax.add_patch(arc)

        # Label
        mid_angle = (theta1 + theta2) / 2
        lx = 1.15 * np.cos(mid_angle)
        ly = 1.15 * np.sin(mid_angle)
        ha = "left" if -np.pi / 2 < mid_angle < np.pi / 2 else "right"
        ax.text(lx, ly, group_names[i], ha=ha, va="center", fontsize=9, fontweight="bold")

    # Draw chords
    max_weight = net.max() if net.max() > 0 else 1.0

    for i in range(n_groups):
        for j in range(n_groups):
            if net[i, j] <= 0:
                continue

            theta_i1, theta_i2 = angles[i]
            theta_j1, theta_j2 = angles[j]

            # Allocate fraction of arc based on weight
            frac_i = net[i, j] / row_sums[i] if row_sums[i] > 0 else 0
            frac_j = net[i, j] / col_sums[j] if col_sums[j] > 0 else 0

            src_start = theta_i1 + (theta_i2 - theta_i1) * 0.1
            src_end = src_start + (theta_i2 - theta_i1) * frac_i * 0.8
            tgt_start = theta_j1 + (theta_j2 - theta_j1) * 0.1
            tgt_end = tgt_start + (theta_j2 - theta_j1) * frac_j * 0.8

            # Draw chord as filled path
            src_mid = (src_start + src_end) / 2
            tgt_mid = (tgt_start + tgt_end) / 2

            # Quadratic bezier through center
            verts = [
                (np.cos(src_mid), np.sin(src_mid)),
                (0, 0),
                (np.cos(tgt_mid), np.sin(tgt_mid)),
            ]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)

            lw = 0.5 + 4.0 * net[i, j] / max_weight
            patch = mpatches.FancyArrowPatch(
                path=path,
                arrowstyle="-",
                color=color_use[i],
                alpha=transparency,
                linewidth=lw * 0.3,
            )
            ax.add_patch(patch)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig
