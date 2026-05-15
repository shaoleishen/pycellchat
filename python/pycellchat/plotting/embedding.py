"""Network embedding visualization."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def net_visual_embedding(
    cc_obj,
    slot_name: str = "netP",
    type: str = "functional",
    color_use: Optional[list[str]] = None,
    point_size: float = 100,
    fig_size: tuple = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """UMAP embedding of signaling pathway networks.

    Each point represents a signaling pathway, colored by cluster.

    Parameters
    ----------
    cc_obj
        CellChat object with netP computed.
    type
        ``"functional"`` or ``"structural"``.
    color_use
        Colors for clusters.
    point_size
        Point size.
    fig_size
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    cc = cc_obj.cc
    netp = cc.get(slot_name)
    if netp is None or "similarity" not in netp:
        raise RuntimeError("Run compute_net_similarity() first")

    similarity = netp["similarity"]
    pathways = netp["pathways"]
    n_pathways = len(pathways)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    # Use UMAP on similarity matrix
    try:
        import umap
        reducer = umap.UMAP(metric="precomputed", n_neighbors=min(5, n_pathways - 1), random_state=42)
        distance = 1 - similarity
        np.fill_diagonal(distance, 0)
        embedding = reducer.fit_transform(distance)
    except ImportError:
        # Fallback: PCA-based 2D embedding
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(similarity)

    # Color by total communication
    scores = netp["prob"].sum(axis=(0, 1)) if "prob" in netp else np.ones(n_pathways)

    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=scores, cmap="YlOrRd", s=point_size, alpha=0.8, edgecolors="gray"
    )

    # Label points
    for i, pw in enumerate(pathways):
        ax.annotate(pw, (embedding[i, 0], embedding[i, 1]),
                     fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                     textcoords="offset points")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Total communication probability")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Network embedding ({type})", fontsize=12)

    fig.tight_layout()
    return fig


def net_visual_embedding_pairwise(
    cc_obj1,
    cc_obj2,
    names: tuple[str, str] = ("Dataset 1", "Dataset 2"),
    fig_size: tuple = (12, 5),
) -> plt.Figure:
    """Compare network embeddings between two datasets.

    Parameters
    ----------
    cc_obj1, cc_obj2
        CellChat objects to compare.
    names
        Names for each dataset.

    Returns
    -------
    matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=fig_size)

    for idx, (cc_obj, name) in enumerate(zip([cc_obj1, cc_obj2], names)):
        try:
            net_visual_embedding(cc_obj, ax=axes[idx])
            axes[idx].set_title(name)
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=axes[idx].transAxes)

    fig.tight_layout()
    return fig
