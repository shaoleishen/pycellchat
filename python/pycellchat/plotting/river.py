"""River/alluvial plot for communication patterns."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def net_analysis_river(
    cc_obj,
    pattern: str = "outgoing",
    slot_name: str = "netP",
    color_use: Optional[list[str]] = None,
    fig_size: tuple = (12, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """River/alluvial plot showing communication patterns.

    Shows how cell groups contribute to different signaling patterns.

    Parameters
    ----------
    cc_obj
        CellChat object with patterns computed.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    slot_name
        Slot name.
    color_use
        Colors for patterns.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    netp = cc.get(slot_name)
    if netp is None or "pattern" not in netp:
        raise RuntimeError("Run identify_communication_patterns() first")

    pattern_data = netp["pattern"].get(pattern)
    if pattern_data is None:
        raise RuntimeError(f"No {pattern} patterns found")

    W = pattern_data["W"]  # K x k (group contributions to patterns)
    H = pattern_data["H"]  # k x n_pathways (pattern loadings)
    group_names = pattern_data["group_names"]
    pathways = pattern_data["pathways"]
    k = pattern_data["k"]

    if color_use is None:
        cmap = plt.cm.Set3
        color_use = [cmap(i / k) for i in range(k)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    ax.axis("off")

    n_groups = len(group_names)
    n_pathways = len(pathways)

    # Layout: groups on left, patterns in middle, pathways on right
    group_x = 0.1
    pattern_x = 0.5
    pathway_x = 0.9

    group_y = np.linspace(0.1, 0.9, n_groups)
    pattern_y = np.linspace(0.1, 0.9, k)
    pathway_y = np.linspace(0.1, 0.9, n_pathways)

    # Draw group nodes
    for i, name in enumerate(group_names):
        ax.plot(group_x, group_y[i], "o", color="steelblue", markersize=15, zorder=3)
        ax.text(group_x - 0.02, group_y[i], name, ha="right", va="center", fontsize=9)

    # Draw pattern nodes
    for p in range(k):
        ax.plot(pattern_x, pattern_y[p], "o", color=color_use[p], markersize=20, zorder=3)
        ax.text(pattern_x, pattern_y[p], f"P{p+1}", ha="center", va="center", fontsize=8, fontweight="bold")

    # Draw pathway nodes
    for j, name in enumerate(pathways):
        ax.plot(pathway_x, pathway_y[j], "o", color="coral", markersize=10, zorder=3)
        ax.text(pathway_x + 0.02, pathway_y[j], name, ha="left", va="center", fontsize=8)

    # Draw group -> pattern flows
    max_w = W.max() if W.max() > 0 else 1.0
    for i in range(n_groups):
        for p in range(k):
            w = W[i, p]
            if w > 0.01:
                lw = 0.5 + 5 * w / max_w
                ax.plot(
                    [group_x, pattern_x], [group_y[i], pattern_y[p]],
                    color=color_use[p], alpha=0.4, linewidth=lw
                )

    # Draw pattern -> pathway flows
    max_h = H.max() if H.max() > 0 else 1.0
    for p in range(k):
        for j in range(min(n_pathways, H.shape[1])):
            h = H[p, j]
            if h > 0.01:
                lw = 0.5 + 5 * h / max_h
                ax.plot(
                    [pattern_x, pathway_x], [pattern_y[p], pathway_y[j]],
                    color=color_use[p], alpha=0.4, linewidth=lw
                )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_title(f"{pattern.capitalize()} communication patterns", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig
