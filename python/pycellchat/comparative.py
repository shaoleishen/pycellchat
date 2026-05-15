"""Comparative analysis: merge, compare, rank across conditions.

Port of CellChat R analysis.R comparative functions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def merge_cell_chat(
    cc_list: list,
    names: Optional[list[str]] = None,
    merge_data: bool = False,
) -> dict:
    """Merge multiple CellChat objects for comparison.

    Parameters
    ----------
    cc_list
        List of CellChat objects.
    names
        Names for each dataset.
    merge_data
        Whether to merge underlying expression data.

    Returns
    -------
    Merged CellChat data dict.
    """
    if names is None:
        names = [f"dataset{i}" for i in range(len(cc_list))]

    merged = {
        "mode": "merged",
        "names": names,
        "net": [],
        "netP": [],
    }

    for cc_obj in cc_list:
        cc = cc_obj.cc
        merged["net"].append(cc.get("net", {}))
        merged["netP"].append(cc.get("netP", {}))

    return merged


def compare_interactions(
    cc_list: list,
    names: Optional[list[str]] = None,
    measure: str = "count",
) -> dict:
    """Compare interaction counts or weights across conditions.

    Parameters
    ----------
    cc_list
        List of CellChat objects.
    names
        Dataset names.
    measure
        ``"count"`` or ``"weight"``.

    Returns
    -------
    Dict with comparison results.
    """
    if names is None:
        names = [f"dataset{i}" for i in range(len(cc_list))]

    results = {"names": names, "measure": measure, "values": []}

    for cc_obj in cc_list:
        cc = cc_obj.cc
        net = cc.get("net", {})
        if measure in net:
            results["values"].append(net[measure])
        else:
            results["values"].append(None)

    return results


def rank_net(
    cc_obj,
    measure: str = "weight",
    mode: str = "single",
) -> list[tuple[str, float]]:
    """Rank signaling pathways by total communication probability.

    Parameters
    ----------
    cc_obj
        CellChat object.
    measure
        ``"weight"`` or ``"count"``.
    mode
        ``"single"`` or ``"comparison"``.

    Returns
    -------
    List of (pathway_name, score) tuples, sorted descending.
    """
    cc = cc_obj.cc
    netp = cc.get("netP", {})
    if "prob" not in netp:
        raise RuntimeError("Run compute_commun_prob_pathway() first")

    prob = netp["prob"]
    pathways = netp["pathways"]

    # Total probability per pathway
    scores = prob.sum(axis=(0, 1))

    # Sort descending
    order = np.argsort(-scores)
    ranked = [(pathways[i], scores[i]) for i in order if scores[i] > 0]

    return ranked


def rank_similarity(cc_obj) -> list[tuple[str, float]]:
    """Rank pathways by network similarity.

    Parameters
    ----------
    cc_obj
        CellChat object.

    Returns
    -------
    List of (pathway_name, similarity_score).
    """
    cc = cc_obj.cc
    netp = cc.get("netP", {})
    similarity = netp.get("similarity")
    if similarity is None:
        raise RuntimeError("Run compute_net_similarity() first")

    pathways = netp["pathways"]
    n = len(pathways)

    # Average similarity per pathway
    scores = similarity.mean(axis=1)
    order = np.argsort(-scores)

    return [(pathways[i], scores[i]) for i in order]


def net_mapping_deg(
    cc_obj,
    deg_df,
    gene_col: str = "gene",
    cluster_col: str = "cluster",
    logfc_col: str = "logFC",
    pval_col: str = "p_val_adj",
) -> dict:
    """Map differentially expressed genes to communication network.

    Parameters
    ----------
    cc_obj
        CellChat object.
    deg_df
        DataFrame with DEG results.
    gene_col
        Column name for gene names.
    cluster_col
        Column name for cluster/group.
    logfc_col
        Column name for log fold change.
    pval_col
        Column name for adjusted p-value.

    Returns
    -------
    Dict mapping pathways to DEGs.
    """
    cc = cc_obj.cc
    netp = cc.get("netP", {})
    lr_sig = cc.get("LR", {}).get("LRsig")

    if lr_sig is None:
        raise RuntimeError("No LR signature found")

    # Get genes involved in each pathway
    pathway_genes = {}
    for _, row in lr_sig.iterrows():
        pathway = row.get("pathway_name", "")
        for col in ["ligand", "receptor"]:
            gene = row.get(col, "")
            if gene:
                pathway_genes.setdefault(pathway, set()).add(gene)

    # Map DEGs to pathways
    deg_set = set(deg_df[gene_col].tolist())
    mapping = {}
    for pathway, genes in pathway_genes.items():
        overlap = genes & deg_set
        if overlap:
            mapping[pathway] = sorted(overlap)

    return mapping
