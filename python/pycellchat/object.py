"""CellChat class: AnnData-native cell-cell communication analysis."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata import AnnData
except ImportError:
    raise ImportError("anndata is required: pip install anndata")

logger = logging.getLogger(__name__)


class CellChat:
    """Cell-cell communication analysis wrapper around AnnData.

    All results are stored in ``adata.uns['cellchat']`` for seamless
    integration with the scanpy ecosystem.

    Parameters
    ----------
    adata
        Annotated data matrix with gene expression.
    group_by
        Column in ``adata.obs`` containing cell group labels.
    datatype
        Data type: ``"RNA"`` (default) or ``"spatial"``.
    coordinates
        Spatial coordinates array (n_cells x 2). Required for spatial data.
    spatial_factors
        Spatial factors dict with ``ratio`` and ``tol`` keys.
    """

    def __init__(
        self,
        adata: AnnData,
        group_by: str,
        datatype: str = "RNA",
        coordinates: Optional[np.ndarray] = None,
        spatial_factors: Optional[dict] = None,
    ):
        if group_by not in adata.obs.columns:
            raise KeyError(f"Column '{group_by}' not found in adata.obs")

        self.adata = adata
        self.group_by = group_by
        self.datatype = datatype

        # Initialize the cellchat namespace in uns
        if "cellchat" not in adata.uns:
            adata.uns["cellchat"] = {}

        cc = adata.uns["cellchat"]
        cc.setdefault("options", {})
        cc["options"]["mode"] = "single"
        cc["options"]["datatype"] = datatype

        # Store cell identities as categorical codes
        groups = adata.obs[group_by]
        self._idents = groups.astype("category")
        self._group_codes = self._idents.cat.codes.values.astype(int)
        self._group_names = list(self._idents.cat.categories)
        self._n_groups = len(self._group_names)

        cc["idents"] = {
            "codes": self._group_codes,
            "names": self._group_names,
            "column": group_by,
        }

        # Auto-detect spatial data from scanpy conventions
        if coordinates is None and "spatial" in adata.obsm:
            coordinates = adata.obsm["spatial"]
            if datatype == "RNA":
                datatype = "spatial"
                self.datatype = datatype
                cc["options"]["datatype"] = datatype
            logger.info("Auto-detected spatial coordinates from adata.obsm['spatial']")

        # Spatial data
        if datatype == "spatial":
            if coordinates is None:
                raise ValueError("coordinates required for spatial data (pass explicitly or store in adata.obsm['spatial'])")
            cc.setdefault("images", {})
            cc["images"]["coordinates"] = np.asarray(coordinates)
            if spatial_factors is not None:
                cc["images"]["spatial_factors"] = spatial_factors

    @property
    def n_groups(self) -> int:
        return self._n_groups

    @property
    def group_names(self) -> list[str]:
        return self._group_names

    @property
    def cc(self) -> dict:
        """Direct access to adata.uns['cellchat']."""
        return self.adata.uns["cellchat"]

    def set_db(self, species: str = "human") -> None:
        """Load CellChatDB for the given species."""
        from pycellchat.database import load_cellchatdb
        self.cc["DB"] = load_cellchatdb(species)
        logger.info(f"Loaded CellChatDB for {species}")

    def subset_data(self, features: Optional[list[str]] = None, keep_sparse: bool = False) -> None:
        """Subset signaling genes from the expression data.

        Stores the subset in adata.uns['cellchat']['data.signaling'].

        Parameters
        ----------
        features
            Specific genes to use. If None, use all signaling genes from DB.
        keep_sparse
            If True, keep data as sparse CSR matrix (genes x cells).
            If False (default), densify with .toarray().
        """
        db = self.cc.get("DB")
        if db is None:
            raise RuntimeError("Run set_db() first to load CellChatDB")

        # Collect all genes mentioned in the database
        interaction = db["interaction"]
        lr_genes = set()
        for col in ["ligand", "receptor"]:
            lr_genes.update(interaction[col].dropna().unique())

        # Also include complex subunits
        if "complex" in db:
            complex_df = db["complex"]
            for col in complex_df.columns:
                if col != "name":
                    vals = complex_df[col].dropna()
                    vals = vals[vals != ""]
                    lr_genes.update(vals.unique())

        # Also include cofactor genes
        if "cofactor" in db:
            cofactor_df = db["cofactor"]
            for col in cofactor_df.columns:
                if col != "name":
                    vals = cofactor_df[col].dropna()
                    vals = vals[vals != ""]
                    lr_genes.update(vals.unique())

        # Intersect with available genes
        available_genes = set(self.adata.var_names.tolist())
        signaling_genes = sorted(lr_genes & available_genes)

        if not signaling_genes:
            raise ValueError("No signaling genes found in adata.var_names")

        # Subset expression data (AnnData X is cells x genes, we store genes x cells)
        gene_mask = self.adata.var_names.isin(signaling_genes)
        if keep_sparse and sparse.issparse(self.adata.X):
            # Keep sparse: cells x genes → genes x cells (CSR)
            data_signaling = sparse.csr_matrix(self.adata.X[:, gene_mask]).T.tocsr()
        elif sparse.issparse(self.adata.X):
            data_signaling = self.adata.X[:, gene_mask].toarray().T
        else:
            data_signaling = self.adata.X[:, gene_mask].copy().T

        self.cc["data.signaling"] = data_signaling
        self.cc["signaling_genes"] = self.adata.var_names[gene_mask].tolist()

        # Build gene index for fast lookup
        self.cc["gene_index"] = {
            gene: idx for idx, gene in enumerate(self.cc["signaling_genes"])
        }

        logger.info(
            f"Subset {len(signaling_genes)} signaling genes from {self.adata.n_vars} total"
        )

    def normalize(self, scale_factor: float = 10000.0, do_log: bool = True) -> None:
        """Normalize raw count data and store in adata.uns['cellchat']['data.norm'].

        The normalized data is stored in genes x cells orientation.
        """
        from pycellchat.preprocessing import normalize_data

        # AnnData X is cells x genes; normalize_data expects genes x cells
        if sparse.issparse(self.adata.X):
            data_raw = self.adata.X.T.tocsr()
        else:
            data_raw = self.adata.X.T

        self.cc["data.norm"] = normalize_data(data_raw, scale_factor, do_log)
        logger.info("Normalized data stored in adata.uns['cellchat']['data.norm']")

    def save(self, path: str) -> None:
        """Save CellChat results to a directory.

        Parameters
        ----------
        path
            Output directory path.
        """
        from pycellchat.io import save_cellchat
        save_cellchat(self, path)

    def set_spatial_params(
        self,
        interaction_range: float = 250.0,
        contact_range: float = 10.0,
        ratio: float = 1.0,
        tol: Optional[float] = None,
        scale_distance: float = 0.01,
    ) -> None:
        """Set spatial parameters for spatial analysis.

        Parameters
        ----------
        interaction_range
            Maximum diffusion range of ligands (microns).
        contact_range
            Maximum range for contact-dependent signaling (microns).
        ratio
            Pixel-to-micron conversion factor.
        tol
            Distance tolerance. Defaults to contact_range / 2.
        scale_distance
            Normalization factor for distance-to-probability conversion.
        """
        if tol is None:
            tol = contact_range / 2.0

        if "images" not in self.cc:
            self.cc["images"] = {}

        self.cc["images"]["spatial_factors"] = {
            "interaction_range": interaction_range,
            "contact_range": contact_range,
            "ratio": ratio,
            "tol": tol,
            "scale_distance": scale_distance,
        }

    def __repr__(self) -> str:
        n = self.adata.n_obs
        k = self._n_groups
        return f"CellChat with {n} cells in {k} groups ({self.group_by})"
