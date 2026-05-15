"""Enrichment analysis and DEG mapping.

Port of CellChat R enrichment functions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_enrichment_score(
    cc_obj,
    signaling: Optional[str] = None,
) -> dict[str, float]:
    """Compute enrichment score for signaling pathways.

    The enrichment score measures how concentrated the signaling
    is among specific cell groups.

    Parameters
    ----------
    cc_obj
        CellChat object.
    signaling
        Specific pathway to compute. If None, compute for all.

    Returns
    -------
    Dict of pathway_name -> enrichment score.
    """
    cc = cc_obj.cc
    netp = cc.get("netP", {})
    if "prob" not in netp:
        raise RuntimeError("Run compute_commun_prob_pathway() first")

    prob = netp["prob"]
    pathways = netp["pathways"]
    scores = {}

    for pw_idx, pw_name in enumerate(pathways):
        if signaling and pw_name != signaling:
            continue

        pw_prob = prob[:, :, pw_idx]
        total = pw_prob.sum()
        if total == 0:
            scores[pw_name] = 0.0
            continue

        # Enrichment: max cell group contribution / uniform contribution
        row_sums = pw_prob.sum(axis=1)
        col_sums = pw_prob.sum(axis=0)
        max_contribution = max(row_sums.max(), col_sums.max())
        uniform = total / (pw_prob.shape[0] * 2)

        scores[pw_name] = max_contribution / uniform if uniform > 0 else 0.0

    return scores


def extract_enriched_lr(
    cc_obj,
    signaling: str,
    thresh: float = 0.05,
) -> pd.DataFrame:
    """Extract enriched LR pairs for a given pathway.

    Parameters
    ----------
    cc_obj
        CellChat object.
    signaling
        Pathway name.
    thresh
        p-value threshold.

    Returns
    -------
    DataFrame of enriched LR pairs.
    """
    cc = cc_obj.cc
    net = cc.get("net", {})
    lr_sig = cc.get("LR", {}).get("LRsig")

    if lr_sig is None or "prob" not in net:
        raise RuntimeError("Run compute_commun_prob() first")

    prob = net["prob"]
    pval = net["pval"]

    # Find LR pairs belonging to this pathway
    pathway_mask = lr_sig["pathway_name"] == signaling
    lr_indices = np.where(pathway_mask)[0]

    if len(lr_indices) == 0:
        return pd.DataFrame()

    results = []
    group_names = cc["idents"]["names"]

    for lr_idx in lr_indices:
        lr_prob = prob[:, :, lr_idx]
        lr_pval = pval[:, :, lr_idx]

        # Find significant source-target pairs
        sig_mask = lr_pval < thresh
        if not sig_mask.any():
            continue

        lr_info = lr_sig.iloc[lr_idx]
        for src in range(prob.shape[0]):
            for tgt in range(prob.shape[1]):
                if sig_mask[src, tgt]:
                    results.append({
                        "interaction_name": lr_info.get("interaction_name", ""),
                        "ligand": lr_info.get("ligand", ""),
                        "receptor": lr_info.get("receptor", ""),
                        "pathway_name": signaling,
                        "source": group_names[src],
                        "target": group_names[tgt],
                        "prob": lr_prob[src, tgt],
                        "pval": lr_pval[src, tgt],
                    })

    return pd.DataFrame(results) if results else pd.DataFrame()


def extract_enriched_signaling(
    cc_obj,
    thresh: float = 0.05,
) -> list[str]:
    """Extract pathways with significant communication.

    Parameters
    ----------
    cc_obj
        CellChat object.
    thresh
        p-value threshold.

    Returns
    -------
    List of significant pathway names.
    """
    cc = cc_obj.cc
    net = cc.get("net", {})
    lr_sig = cc.get("LR", {}).get("LRsig")

    if lr_sig is None or "pval" not in net:
        raise RuntimeError("Run compute_commun_prob() first")

    pval = net["pval"]

    # For each pathway, check if any LR pair has significant p-value
    pathways = lr_sig["pathway_name"].unique()
    sig_pathways = []

    for pathway in pathways:
        pathway_mask = lr_sig["pathway_name"] == pathway
        lr_indices = np.where(pathway_mask)[0]

        for lr_idx in lr_indices:
            if (pval[:, :, lr_idx] < thresh).any():
                sig_pathways.append(pathway)
                break

    return sorted(sig_pathways)
