"""I/O utilities: create CellChat from AnnData, save/load results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from anndata import AnnData
except ImportError:
    raise ImportError("anndata is required: pip install anndata")

from pycellchat.object import CellChat

logger = logging.getLogger(__name__)


def from_anndata(
    adata: AnnData,
    group_by: str,
    datatype: str = "RNA",
    coordinates: np.ndarray | None = None,
    spatial_factors: dict | None = None,
    species: str = "human",
) -> CellChat:
    """Create a CellChat object from an AnnData.

    This is a convenience function that creates a CellChat object
    and optionally loads the database.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group_by
        Column in adata.obs for cell grouping.
    datatype
        ``"RNA"`` or ``"spatial"``.
    coordinates
        Spatial coordinates (required for spatial data).
    spatial_factors
        Spatial factors dict.
    species
        Species for CellChatDB (default: ``"human"``).

    Returns
    -------
    CellChat object.
    """
    cc = CellChat(
        adata,
        group_by=group_by,
        datatype=datatype,
        coordinates=coordinates,
        spatial_factors=spatial_factors,
    )
    cc.set_db(species)
    return cc


def save_cellchat(cc_obj, path: str) -> None:
    """Save CellChat results to a directory.

    Saves network matrices (CSV), pathway results (JSON),
    and summary (JSON) to the specified directory.

    Parameters
    ----------
    cc_obj
        CellChat object with results.
    path
        Output directory path.
    """
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)

    cc = cc_obj.cc

    # Save summary
    summary = {
        "n_groups": cc_obj.n_groups,
        "group_names": cc_obj.group_names,
        "group_by": cc["idents"]["column"],
        "datatype": cc["options"].get("datatype", "RNA"),
    }

    if "options" in cc and "parameter" in cc["options"]:
        summary["params"] = cc["options"]["parameter"]
    if "options" in cc and "run.time" in cc["options"]:
        summary["run_time"] = cc["options"]["run.time"]

    # Save network matrices
    if "net" in cc:
        net = cc["net"]
        if "count" in net:
            count_df = pd.DataFrame(
                net["count"],
                index=cc_obj.group_names,
                columns=cc_obj.group_names,
            )
            count_df.to_csv(outdir / "net_count.csv")
            summary["total_interactions"] = int(net["count"].sum())

        if "weight" in net:
            weight_df = pd.DataFrame(
                net["weight"],
                index=cc_obj.group_names,
                columns=cc_obj.group_names,
            )
            weight_df.to_csv(outdir / "net_weight.csv")
            summary["mean_weight"] = float(net["weight"].mean())

        if "prob" in net:
            summary["n_lr_pairs"] = int(net["prob"].shape[2])

    # Save pathway results
    if "netP" in cc:
        netp = cc["netP"]
        if "pathways" in netp:
            summary["n_pathways"] = len(netp["pathways"])
            summary["pathways"] = netp["pathways"]

            # Save pathway probabilities
            if "prob" in netp:
                pw_data = []
                for i, pw in enumerate(netp["pathways"]):
                    mat = netp["prob"][:, :, i]
                    pw_data.append({
                        "pathway": pw,
                        "total_prob": float(mat.sum()),
                        "mean_prob": float(mat.mean()),
                        "max_prob": float(mat.max()),
                    })
                pw_df = pd.DataFrame(pw_data).sort_values("total_prob", ascending=False)
                pw_df.to_csv(outdir / "pathway_summary.csv", index=False)

    # Save LR pair info
    if "LR" in cc and "LRsig" in cc["LR"]:
        lr_sig = cc["LR"]["LRsig"]
        if isinstance(lr_sig, pd.DataFrame):
            lr_sig.to_csv(outdir / "lr_pairs.csv", index=False)

    # Save summary JSON
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"CellChat results saved to {outdir}")


def load_cellchat_results(path: str) -> dict:
    """Load CellChat results from a saved directory.

    Parameters
    ----------
    path
        Directory path containing saved results.

    Returns
    -------
    Dict with loaded results.
    """
    indir = Path(path)
    if not indir.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    results = {}

    # Load summary
    summary_path = indir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results["summary"] = json.load(f)

    # Load network matrices
    for name in ["net_count", "net_weight"]:
        csv_path = indir / f"{name}.csv"
        if csv_path.exists():
            results[name] = pd.read_csv(csv_path, index_col=0)

    # Load pathway summary
    pw_path = indir / "pathway_summary.csv"
    if pw_path.exists():
        results["pathway_summary"] = pd.read_csv(pw_path)

    # Load LR pairs
    lr_path = indir / "lr_pairs.csv"
    if lr_path.exists():
        results["lr_pairs"] = pd.read_csv(lr_path)

    logger.info(f"Loaded CellChat results from {indir}")
    return results
