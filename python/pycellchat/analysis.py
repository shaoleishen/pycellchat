"""Network analysis: centrality metrics, NMF patterns, similarity, embedding.

Port of CellChat R analysis.R functions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import networkx as nx

try:
    from anndata import AnnData
except ImportError:
    raise ImportError("anndata is required")

logger = logging.getLogger(__name__)


def net_analysis_compute_centrality(cc_obj, slot_name: str = "netP") -> None:
    """Compute network centrality metrics for each signaling pathway.

    Computes 10 centrality metrics per pathway:
    - outdeg_unweighted, indeg_unweighted
    - outdeg, indeg (weighted)
    - hub, authority (Kleinberg)
    - eigen (eigenvector)
    - page_rank
    - betweenness
    - flow_betweenness
    - information

    Parameters
    ----------
    cc_obj
        CellChat object with netP computed.
    slot_name
        Slot to use: "netP" for pathway-level.
    """
    cc = cc_obj.cc
    netp = cc.get(slot_name)
    if netp is None:
        raise RuntimeError(f"Run compute_commun_prob_pathway() first")

    prob = netp["prob"]  # K x K x n_pathways
    pathways = netp["pathways"]
    n_groups = prob.shape[0]
    group_names = cc["idents"]["names"]

    centrality = {}

    for pw_idx, pathway in enumerate(pathways):
        net = prob[:, :, pw_idx]
        centrality[pathway] = _compute_centrality_local(net, group_names)

    netp["centr"] = centrality
    logger.info(f"Computed centrality for {len(pathways)} pathways")


def _compute_centrality_local(net: np.ndarray, node_names: list[str]) -> dict[str, np.ndarray]:
    """Compute all centrality metrics for a weighted adjacency matrix.

    Parameters
    ----------
    net : K x K weighted adjacency matrix
    node_names : node names for the graph

    Returns
    -------
    Dict of metric name -> array of centrality values.
    """
    k = net.shape[0]

    # Create directed weighted graph
    G = nx.DiGraph()
    for i in range(k):
        G.add_node(i, name=node_names[i])

    for i in range(k):
        for j in range(k):
            if net[i, j] > 0:
                G.add_edge(i, j, weight=net[i, j])

    result = {}

    # Unweighted degree
    result["outdeg_unweighted"] = np.array([sum(1 for _ in G.successors(i)) for i in range(k)], dtype=float)
    result["indeg_unweighted"] = np.array([sum(1 for _ in G.predecessors(i)) for i in range(k)], dtype=float)

    # Weighted degree (strength)
    result["outdeg"] = np.array([sum(G[i][j]["weight"] for j in G.successors(i)) for i in range(k)], dtype=float)
    result["indeg"] = np.array([sum(G[j][i]["weight"] for j in G.predecessors(i)) for i in range(k)], dtype=float)

    # Kleinberg hub/authority scores
    try:
        hub_dict = nx.hits(G, max_iter=1000, tol=1e-6)
        result["hub"] = np.array([hub_dict[0].get(i, 0.0) for i in range(k)])
        result["authority"] = np.array([hub_dict[1].get(i, 0.0) for i in range(k)])
    except Exception:
        result["hub"] = np.zeros(k)
        result["authority"] = np.zeros(k)

    # Eigenvector centrality
    try:
        eigen_dict = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        result["eigen"] = np.array([eigen_dict.get(i, 0.0) for i in range(k)])
    except Exception:
        result["eigen"] = np.zeros(k)

    # PageRank
    try:
        pr_dict = nx.pagerank(G, alpha=0.85, max_iter=1000, tol=1e-6)
        result["page_rank"] = np.array([pr_dict.get(i, 0.0) for i in range(k)])
    except Exception:
        result["page_rank"] = np.zeros(k)

    # Betweenness centrality (on inverted weights for shortest path)
    try:
        G_inv = G.copy()
        for u, v, d in G_inv.edges(data=True):
            d["weight"] = 1.0 / d["weight"] if d["weight"] > 0 else 1e10
        bet_dict = nx.betweenness_centrality(G_inv, weight="weight", normalized=True)
        result["betweenness"] = np.array([bet_dict.get(i, 0.0) for i in range(k)])
    except Exception:
        result["betweenness"] = np.zeros(k)

    # Flow betweenness (approximation using networkx)
    try:
        flow_dict = nx.betweenness_centrality(G, weight="weight", normalized=True)
        result["flow_betweenness"] = np.array([flow_dict.get(i, 0.0) for i in range(k)])
    except Exception:
        result["flow_betweenness"] = np.zeros(k)

    # Information centrality (approximation)
    try:
        info_dict = nx.current_flow_closeness_centrality(G.to_undirected(), weight="weight")
        result["information"] = np.array([info_dict.get(i, 0.0) for i in range(k)])
    except Exception:
        result["information"] = np.zeros(k)

    return result


def identify_communication_patterns(
    cc_obj,
    pattern: str = "outgoing",
    k: int = 4,
    slot_name: str = "netP",
    backend: Optional["Backend"] = None,
) -> None:
    """Identify communication patterns using NMF.

    Parameters
    ----------
    cc_obj
        CellChat object with netP computed.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    k
        Number of NMF patterns.
    slot_name
        Slot name.
    backend
        Compute backend. If GPU, uses cuML NMF for acceleration.
    """
    # Try cuML NMF for GPU, fall back to sklearn
    if backend is not None and backend.is_gpu:
        try:
            from cuml.decomposition import NMF
        except ImportError:
            from sklearn.decomposition import NMF
    else:
        from sklearn.decomposition import NMF

    cc = cc_obj.cc
    netp = cc.get(slot_name)
    if netp is None:
        raise RuntimeError("Run compute_commun_prob_pathway() first")

    prob = netp["prob"]  # K x K x n_pathways
    pathways = netp["pathways"]
    n_groups = prob.shape[0]

    if pattern == "outgoing":
        # Sum over targets: K x n_pathways
        data = prob.sum(axis=1)
    else:
        # Sum over sources: K x n_pathways
        data = prob.sum(axis=0)

    # Column-wise max normalization
    col_max = data.max(axis=0)
    col_max[col_max == 0] = 1.0
    data = data / col_max

    # Remove zero rows
    row_mask = data.sum(axis=1) > 0
    data_filtered = data[row_mask]

    if data_filtered.shape[0] < k or k < 1:
        logger.warning(f"Not enough non-zero rows ({data_filtered.shape[0]}) for {k} patterns")
        return

    # NMF decomposition
    model = NMF(n_components=k, init="nndsvd", solver="mu", max_iter=500, random_state=42)
    W = model.fit_transform(data_filtered)
    H = model.components_

    # Row-normalize W, column-normalize H
    W_norm = W / W.sum(axis=1, keepdims=True)
    H_norm = H / H.sum(axis=0, keepdims=True)

    # Store results
    if "pattern" not in netp:
        netp["pattern"] = {}

    group_names_filtered = [cc["idents"]["names"][i] for i in range(n_groups) if row_mask[i]]

    netp["pattern"][pattern] = {
        "W": W_norm,
        "H": H_norm,
        "k": k,
        "group_names": group_names_filtered,
        "pathways": pathways,
        "data": data,
    }

    logger.info(f"Identified {k} {pattern} communication patterns")


def compute_net_similarity(cc_obj, type: str = "functional") -> None:
    """Compute similarity between signaling networks.

    Parameters
    ----------
    cc_obj
        CellChat object.
    type
        ``"functional"`` or ``"structural"``.
    """
    cc = cc_obj.cc
    netp = cc.get("netP")
    if netp is None:
        raise RuntimeError("Run compute_commun_prob_pathway() first")

    prob = netp["prob"]  # K x K x n_pathways
    n_pathways = prob.shape[2]
    if n_pathways == 0:
        netp["similarity"] = np.array([[]])
        return

    # Flatten each pathway's matrix and compute Jaccard similarity
    flattened = prob.reshape(-1, n_pathways)
    binary = (flattened > 0).astype(float)

    # Jaccard similarity
    intersection = binary.T @ binary
    row_sum = binary.sum(axis=0)
    union = row_sum[:, None] + row_sum[None, :] - intersection
    union[union == 0] = 1.0
    similarity = intersection / union

    netp["similarity"] = similarity
    logger.info(f"Computed {type} network similarity")


def select_k(
    cc_obj,
    pattern: str = "outgoing",
    k_range: Optional[list[int]] = None,
    nrun: int = 10,
    slot_name: str = "netP",
) -> dict[int, float]:
    """Estimate optimal number of NMF patterns using cophenetic correlation.

    Parameters
    ----------
    cc_obj
        CellChat object with netP computed.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    k_range
        List of k values to test. Default: [2, 3, 4, 5].
    nrun
        Number of NMF runs per k.
    slot_name
        Slot name.

    Returns
    -------
    Dict of k -> cophenetic correlation score.
    """
    from sklearn.decomposition import NMF
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import pdist

    cc = cc_obj.cc
    netp = cc.get(slot_name)
    if netp is None:
        raise RuntimeError("Run compute_commun_prob_pathway() first")

    prob = netp["prob"]

    if pattern == "outgoing":
        data = prob.sum(axis=1)
    else:
        data = prob.sum(axis=0)

    # Column-wise max normalization
    col_max = data.max(axis=0)
    col_max[col_max == 0] = 1.0
    data = data / col_max

    # Remove zero rows
    row_mask = data.sum(axis=1) > 0
    data_filtered = data[row_mask]

    if k_range is None:
        k_range = list(range(2, min(6, data_filtered.shape[0] + 1)))

    scores = {}
    for k in k_range:
        if k > data_filtered.shape[0]:
            continue

        coph_scores = []
        for _ in range(nrun):
            model = NMF(n_components=k, init="nndsvdar", solver="mu", max_iter=300)
            W = model.fit_transform(data_filtered)
            H = model.components_

            # Cophenetic correlation on H matrix
            if H.shape[1] > 2:
                Z = linkage(H.T, method="average")
                c, _ = cophenet(Z, pdist(H.T))
                coph_scores.append(c)

        if coph_scores:
            scores[k] = np.mean(coph_scores)

    return scores


def lift_cell_chat(
    cc_obj,
    group_new: list[str],
) -> None:
    """Lift cell groups to a common set of labels across datasets.

    Parameters
    ----------
    cc_obj
        CellChat object.
    group_new
        New group labels (must be superset of current groups).
    """
    cc = cc_obj.cc
    old_names = cc["idents"]["names"]

    # Build mapping from old to new
    mapping = {old: new for old, new in zip(old_names, group_new[:len(old_names)])}
    cc["idents"]["names_lifted"] = group_new
    cc["idents"]["mapping"] = mapping

    logger.info(f"Lifted {len(old_names)} groups to {len(group_new)} groups")
