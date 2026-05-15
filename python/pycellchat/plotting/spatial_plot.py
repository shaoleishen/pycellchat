"""Spatial visualization functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def spatial_dim_plot(
    cc_obj,
    color_by: str = "cell_type",
    title: str = "",
    point_size: float = 20,
    color_use: Optional[dict] = None,
    fig_size: tuple = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Spatial dimension plot colored by cell type.

    Parameters
    ----------
    cc_obj
        CellChat object with spatial data.
    color_by
        Column in adata.obs to color by.
    title
        Plot title.
    point_size
        Point size.
    color_use
        Dict mapping category -> color.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    if "images" not in cc or "coordinates" not in cc["images"]:
        raise RuntimeError("No spatial coordinates available")

    coords = cc["images"]["coordinates"]
    labels = cc_obj.adata.obs[color_by].astype("category")
    categories = list(labels.cat.categories)

    if color_use is None:
        cmap = plt.cm.Set2
        color_use = {cat: cmap(i / len(categories)) for i, cat in enumerate(categories)}

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    for cat in categories:
        mask = labels == cat
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_use.get(cat, "gray")],
            s=point_size, label=cat, alpha=0.7, edgecolors="none"
        )

    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title("Spatial cell types", fontsize=12)

    fig.tight_layout()
    return fig


def spatial_feature_plot(
    cc_obj,
    features: list[str],
    title: str = "",
    cmap: str = "viridis",
    point_size: float = 20,
    ncols: int = 3,
    fig_size: Optional[tuple] = None,
) -> plt.Figure:
    """Spatial plot colored by gene expression.

    Parameters
    ----------
    cc_obj
        CellChat object with spatial data.
    features
        Gene names to plot.
    title
        Plot title.
    cmap
        Colormap.
    point_size
        Point size.
    ncols
        Number of columns.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    if "images" not in cc or "coordinates" not in cc["images"]:
        raise RuntimeError("No spatial coordinates available")

    coords = cc["images"]["coordinates"]
    adata = cc_obj.adata

    n_features = len(features)
    nrows = (n_features + ncols - 1) // ncols
    if fig_size is None:
        fig_size = (5 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, gene in enumerate(features):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if gene in adata.var_names:
            values = adata[:, gene].X
            if hasattr(values, "toarray"):
                values = values.toarray().flatten()
            else:
                values = values.flatten()
        else:
            ax.text(0.5, 0.5, f"{gene}\nnot found", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=values, cmap=cmap, s=point_size, alpha=0.8, edgecolors="none"
        )

        ax.set_title(gene, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    # Hide unused axes
    for idx in range(n_features, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def spatial_network_plot(
    cc_obj,
    signaling: Optional[str] = None,
    title: str = "",
    color_use: Optional[list[str]] = None,
    point_size: float = 20,
    edge_width_scale: float = 1.0,
    fig_size: tuple = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot spatial network overlay on coordinates.

    Parameters
    ----------
    cc_obj
        CellChat object with spatial data and network computed.
    signaling
        Pathway to show.
    title
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    group_names = cc["idents"]["names"]
    n_groups = len(group_names)

    if "images" not in cc or "coordinates" not in cc["images"]:
        raise RuntimeError("No spatial coordinates available")

    coords = cc["images"]["coordinates"]
    groups = cc["idents"]["codes"]

    # Get network
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

    # Draw cells
    for g in range(n_groups):
        mask = groups == g
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_use[g]], s=point_size, label=group_names[g],
            alpha=0.7, edgecolors="none"
        )

    # Draw group-level network edges (between group centroids)
    centroids = np.zeros((n_groups, 2))
    for g in range(n_groups):
        mask = groups == g
        if mask.any():
            centroids[g] = coords[mask].mean(axis=0)

    max_weight = net.max() if net.max() > 0 else 1.0
    for i in range(n_groups):
        for j in range(n_groups):
            if net[i, j] > 0:
                lw = 0.5 + edge_width_scale * 3 * net[i, j] / max_weight
                ax.annotate(
                    "", xy=centroids[j], xytext=centroids[i],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color_use[i],
                        lw=lw,
                        alpha=0.5,
                    ),
                )

    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title, fontsize=12)
    elif signaling:
        ax.set_title(f"Spatial network: {signaling}", fontsize=12)
    else:
        ax.set_title("Spatial communication network", fontsize=12)

    fig.tight_layout()
    return fig
