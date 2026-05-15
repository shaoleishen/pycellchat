"""Heatmap for cell-cell communication."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def net_visual_heatmap(
    cc_obj,
    signaling: Optional[str] = None,
    measure: str = "weight",
    title: str = "",
    color_use: str = "RdBu_r",
    fig_size: tuple = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw a heatmap of communication probability or network.

    Parameters
    ----------
    cc_obj
        CellChat object.
    signaling
        Pathway name. If None, use aggregated network.
    measure
        ``"weight"`` or ``"count"``.
    title
        Plot title.
    color_use
        Colormap name.
    fig_size
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]

    if signaling and "netP" in cc:
        pathways = cc["netP"].get("pathways", [])
        if signaling in pathways:
            pw_idx = pathways.index(signaling)
            data = cc["netP"]["prob"][:, :, pw_idx]
        else:
            raise ValueError(f"Pathway '{signaling}' not found")
    elif "net" in cc and measure in cc["net"]:
        data = cc["net"][measure]
    else:
        raise RuntimeError("No network data available")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    sns.heatmap(
        data,
        xticklabels=group_names,
        yticklabels=group_names,
        cmap=color_use,
        ax=ax,
        annot=True if data.shape[0] <= 10 else False,
        fmt=".2f" if data.shape[0] <= 10 else "",
        linewidths=0.5,
    )

    ax.set_xlabel("Target")
    ax.set_ylabel("Source")

    if title:
        ax.set_title(title, fontsize=12)
    elif signaling:
        ax.set_title(f"Pathway: {signaling}", fontsize=12)
    else:
        ax.set_title(f"Network {measure}", fontsize=12)

    fig.tight_layout()
    return fig
