"""Hierarchy plots for cell-cell communication."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def net_visual_hierarchy1(
    cc_obj,
    signaling: Optional[str] = None,
    title: str = "",
    color_use: Optional[list[str]] = None,
    vertex_size: Optional[list[float]] = None,
    edge_width_max: float = 5.0,
    fig_size: tuple = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Hierarchy plot showing source -> target communication.

    Sources on the left, targets on the right, edges between them.

    Parameters
    ----------
    cc_obj
        CellChat object.
    signaling
        Pathway to show.
    title
        Plot title.
    color_use
        Colors for groups.
    vertex_size
        Node sizes.
    edge_width_max
        Maximum edge width.
    fig_size
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]
    n_groups = len(group_names)

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

    ax.axis("off")

    # Position: sources on left (x=0), targets on right (x=1)
    src_y = np.linspace(0.1, 0.9, n_groups)
    tgt_y = np.linspace(0.1, 0.9, n_groups)

    # Draw source nodes
    src_totals = net.sum(axis=1)
    tgt_totals = net.sum(axis=0)
    max_total = max(src_totals.max(), tgt_totals.max()) if max(src_totals.max(), tgt_totals.max()) > 0 else 1.0

    for i in range(n_groups):
        size = 0.02 + 0.05 * src_totals[i] / max_total
        circle = plt.Circle((0.1, src_y[i]), size, color=color_use[i], zorder=3)
        ax.add_patch(circle)
        ax.text(0.02, src_y[i], group_names[i], ha="right", va="center", fontsize=9, fontweight="bold")

        size_tgt = 0.02 + 0.05 * tgt_totals[i] / max_total
        circle = plt.Circle((0.9, tgt_y[i]), size_tgt, color=color_use[i], zorder=3)
        ax.add_patch(circle)
        ax.text(0.98, tgt_y[i], group_names[i], ha="left", va="center", fontsize=9, fontweight="bold")

    # Draw edges
    max_weight = net.max() if net.max() > 0 else 1.0
    for i in range(n_groups):
        for j in range(n_groups):
            if net[i, j] <= 0:
                continue

            lw = 0.5 + edge_width_max * net[i, j] / max_weight
            ax.annotate(
                "", xy=(0.9, tgt_y[j]), xytext=(0.1, src_y[i]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color_use[i],
                    lw=lw,
                    alpha=0.6,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=2,
            )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    elif signaling:
        ax.set_title(f"Pathway: {signaling}", fontsize=12, fontweight="bold")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def net_visual_hierarchy2(
    cc_obj,
    signaling: str,
    title: str = "",
    color_use: Optional[list[str]] = None,
    fig_size: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Hierarchy plot showing LR pair contributions for a pathway.

    Parameters
    ----------
    cc_obj
        CellChat object.
    signaling
        Pathway name.
    title
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]
    lr_sig = cc["LR"]["LRsig"]
    prob = cc["net"]["prob"]

    # Find LR pairs for this pathway
    pathway_mask = lr_sig["pathway_name"] == signaling
    lr_indices = np.where(pathway_mask.values)[0]

    if len(lr_indices) == 0:
        raise ValueError(f"No LR pairs found for pathway '{signaling}'")

    if color_use is None:
        cmap = plt.cm.Set2
        color_use = [cmap(i / len(group_names)) for i in range(len(group_names))]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    # Aggregate by source-target pair for this pathway
    pw_prob = prob[:, :, lr_indices].sum(axis=2)

    # Draw as stacked bars
    n_groups = len(group_names)
    x = np.arange(n_groups)
    width = 0.35

    bottom_out = np.zeros(n_groups)
    bottom_in = np.zeros(n_groups)

    for i in range(n_groups):
        out_vals = pw_prob[i, :]
        in_vals = pw_prob[:, i]

        ax.bar(x - width / 2, out_vals, width, bottom=bottom_out, color=color_use[i], alpha=0.7, label=f"{group_names[i]} send")
        ax.bar(x + width / 2, in_vals, width, bottom=bottom_in, color=color_use[i], alpha=0.4)

        bottom_out += out_vals
        bottom_in += in_vals

    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Communication probability")
    ax.set_xlabel("Cell group")

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f"LR contributions: {signaling}", fontsize=12)

    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
