"""Bar plots for CellChat analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from pycellchat.comparative import rank_net


def net_visual_barplot(
    cc_list: list,
    names: Optional[list[str]] = None,
    measure: str = "count",
    title: str = "",
    fig_size: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar plot comparing interaction counts/weights across conditions.

    Parameters
    ----------
    cc_list
        List of CellChat objects.
    names
        Dataset names.
    measure
        ``"count"`` or ``"weight"``.
    title
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    from pycellchat.comparative import compare_interactions

    if names is None:
        names = [f"dataset{i}" for i in range(len(cc_list))]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    # Compare interactions
    results = compare_interactions(cc_list, names, measure)

    values = []
    for val in results["values"]:
        if val is not None:
            values.append(val.sum())
        else:
            values.append(0)

    bars = ax.bar(names, values, color=plt.cm.Set2(np.linspace(0, 1, len(names))))

    ax.set_ylabel(f"Total {measure}")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Comparison of {measure}")

    fig.tight_layout()
    return fig


def rank_net_plot(
    cc_obj,
    measure: str = "weight",
    top_n: int = 20,
    title: str = "",
    fig_size: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar plot ranking signaling pathways.

    Parameters
    ----------
    cc_obj
        CellChat object.
    top_n
        Number of top pathways to show.
    title
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    ranked = rank_net(cc_obj, measure=measure)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure

    # Take top N
    ranked = ranked[:top_n]
    pathways = [r[0] for r in ranked]
    scores = [r[1] for r in ranked]

    y_pos = range(len(pathways))
    ax.barh(y_pos, scores, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pathways)
    ax.invert_yaxis()
    ax.set_xlabel(f"Total {measure}")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Signaling Pathway Ranking")

    fig.tight_layout()
    return fig
