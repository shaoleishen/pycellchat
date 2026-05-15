"""Dot plot for signaling pathway analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def net_analysis_dot(
    cc_obj,
    pattern: str = "outgoing",
    slot_name: str = "netP",
    title: str = "",
    fig_size: tuple = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Dot plot showing contribution of each cell group to communication patterns.

    Parameters
    ----------
    cc_obj
        CellChat object with patterns computed.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    slot_name
        Slot name.
    title
        Plot title.

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

    W = pattern_data["W"]  # K x k
    group_names = pattern_data["group_names"]
    k = pattern_data["k"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    n_groups = len(group_names)
    x_labels = [f"Pattern {i+1}" for i in range(k)]

    # Create dot plot
    x_pos = []
    y_pos = []
    sizes = []
    colors = []

    for i in range(n_groups):
        for p in range(k):
            x_pos.append(p)
            y_pos.append(i)
            sizes.append(W[i, p] * 500 + 10)
            colors.append(W[i, p])

    scatter = ax.scatter(
        x_pos, y_pos, s=sizes, c=colors,
        cmap="YlOrRd", alpha=0.8, edgecolors="gray", linewidths=0.5
    )

    ax.set_xticks(range(k))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(group_names)
    ax.invert_yaxis()

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Contribution score")

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f"{pattern.capitalize()} pattern contributions", fontsize=12)

    fig.tight_layout()
    return fig


def net_analysis_contribution(
    cc_obj,
    signaling: str,
    title: str = "",
    fig_size: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Show contribution of each LR pair to a signaling pathway.

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
    prob = cc["net"]["prob"]
    lr_sig = cc["LR"]["LRsig"]

    pathway_mask = lr_sig["pathway_name"] == signaling
    lr_indices = np.where(pathway_mask.values)[0]

    if len(lr_indices) == 0:
        raise ValueError(f"No LR pairs for pathway '{signaling}'")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    # Total probability per LR pair
    lr_totals = prob[:, :, lr_indices].sum(axis=(0, 1))
    lr_names = lr_sig.iloc[lr_indices]["interaction_name"].tolist()

    # Sort by total
    order = np.argsort(-lr_totals)
    lr_totals = lr_totals[order]
    lr_names = [lr_names[i] for i in order]

    # Bar plot
    y_pos = range(len(lr_names))
    ax.barh(y_pos, lr_totals, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lr_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total communication probability")

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f"LR contributions to {signaling}", fontsize=12)

    fig.tight_layout()
    return fig
