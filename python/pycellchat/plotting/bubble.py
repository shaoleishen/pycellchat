"""Bubble plot for cell-cell communication."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def net_visual_bubble(
    cc_obj,
    sources_use: Optional[list[str]] = None,
    targets_use: Optional[list[str]] = None,
    signaling: Optional[list[str]] = None,
    thresh: float = 0.05,
    title: str = "",
    fig_size: tuple = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw a bubble plot of communication probability.

    Parameters
    ----------
    cc_obj
        CellChat object with net computed.
    sources_use
        Source cell groups.
    targets_use
        Target cell groups.
    signaling
        Pathways to show.
    thresh
        p-value threshold.
    title
        Plot title.
    fig_size
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]
    n_groups = len(group_names)

    if "net" not in cc or "prob" not in cc["net"]:
        raise RuntimeError("Run compute_commun_prob() first")

    prob = cc["net"]["prob"]
    pval = cc["net"]["pval"]
    lr_sig = cc["LR"]["LRsig"]

    # Filter by signaling
    if signaling:
        pathway_mask = lr_sig["pathway_name"].isin(signaling)
        lr_indices = np.where(pathway_mask)[0]
    else:
        # Show top pathways
        pathway_totals = {}
        for pw in lr_sig["pathway_name"].unique():
            pw_mask = lr_sig["pathway_name"] == pw
            pathway_totals[pw] = prob[:, :, pw_mask.values].sum()
        top_pathways = sorted(pathway_totals, key=pathway_totals.get, reverse=True)[:10]
        pathway_mask = lr_sig["pathway_name"].isin(top_pathways)
        lr_indices = np.where(pathway_mask.values)[0]

    # Aggregate by pathway
    pathways_to_show = lr_sig.iloc[lr_indices]["pathway_name"].unique().tolist()

    # Filter by sources/targets
    src_indices = list(range(n_groups))
    tgt_indices = list(range(n_groups))
    if sources_use:
        src_indices = [i for i, g in enumerate(group_names) if g in sources_use]
    if targets_use:
        tgt_indices = [i for i, g in enumerate(group_names) if g in targets_use]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    # Build data for bubble plot
    x_labels = []
    y_labels = []
    sizes = []
    colors = []

    for pw in pathways_to_show:
        pw_mask = lr_sig["pathway_name"] == pw
        pw_indices = np.where(pw_mask.values)[0]

        for src_i in src_indices:
            for tgt_j in tgt_indices:
                # Average probability across LR pairs in this pathway
                pw_prob = prob[src_i, tgt_j, pw_indices].mean()
                pw_pval = pval[src_i, tgt_j, pw_indices].min()

                if pw_pval < thresh and pw_prob > 0:
                    x_labels.append(f"{group_names[src_i]} -> {group_names[tgt_j]}")
                    y_labels.append(pw)
                    sizes.append(pw_prob)
                    colors.append(-np.log10(pw_pval + 1e-300))

    if not sizes:
        ax.text(0.5, 0.5, "No significant interactions", ha="center", va="center", transform=ax.transAxes)
        return fig

    sizes = np.array(sizes)
    colors = np.array(colors)

    # Normalize sizes for bubble area
    size_norm = sizes / sizes.max() * 300 + 20

    # Create scatter plot
    unique_x = list(dict.fromkeys(x_labels))
    unique_y = list(dict.fromkeys(y_labels))

    x_pos = [unique_x.index(x) for x in x_labels]
    y_pos = [unique_y.index(y) for y in y_labels]

    scatter = ax.scatter(
        x_pos, y_pos, s=size_norm, c=colors,
        cmap="YlOrRd", alpha=0.7, edgecolors="gray", linewidths=0.5
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("-log10(p-value)")

    ax.set_xticks(range(len(unique_x)))
    ax.set_xticklabels(unique_x, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(unique_y)))
    ax.set_yticklabels(unique_y, fontsize=9)

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title("Communication Probability", fontsize=12)

    fig.tight_layout()
    return fig
