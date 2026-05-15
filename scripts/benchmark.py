"""Benchmark pycellchat with large synthetic datasets.

3 datasets with 3000 genes and 14 cell types:
1. ~100K cells (single-cell RNA-seq)
2. ~1M cells (single-cell RNA-seq)
3. ~50K cells (spatial transcriptomics)

Uses real CellChatDB gene names to maximize signaling gene coverage (~2000).
"""

import gc
import time
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")


def timer(label):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            print(f"  {label}: {self.elapsed:.2f}s")
    return Timer()


def get_signaling_gene_names(species="human", n_genes=3000):
    """Extract all signaling gene names from CellChatDB, pad to n_genes."""
    from pycellchat.database import load_cellchatdb
    db = load_cellchatdb(species)
    interaction = db["interaction"]
    complex_df = db["complex"]
    cofactor_df = db["cofactor"]

    genes = []
    # Ligands and receptors
    for col in ["ligand", "receptor"]:
        for g in interaction[col].dropna().unique():
            if g not in genes:
                genes.append(g)

    # Complex subunits
    for col in complex_df.columns:
        if col == "name":
            continue
        for g in complex_df[col].dropna().unique():
            if g != "" and g not in genes:
                genes.append(g)

    # Cofactors
    for col in cofactor_df.columns:
        if col == "name":
            continue
        for g in cofactor_df[col].dropna().unique():
            if g != "" and g not in genes:
                genes.append(g)

    # Filter out non-gene entries (complex names with special chars)
    clean_genes = [g for g in genes if g.isalpha() or (g.isalnum() and not g[0].isdigit())]

    # Pad with synthetic genes if needed
    while len(clean_genes) < n_genes:
        clean_genes.append(f"SYNTH{len(clean_genes)}")

    return clean_genes[:n_genes]


def generate_structured_dataset(n_cells, n_cell_types=14, n_genes=3000, seed=42):
    """Generate synthetic data with structured LR expression patterns."""
    np.random.seed(seed)

    # Get real signaling gene names from CellChatDB
    gene_names = get_signaling_gene_names("human", n_genes)

    # Cell type labels
    cell_types = [f"CT{i:02d}" for i in range(n_cell_types)]
    labels = np.random.choice(cell_types, size=n_cells)

    # Generate base expression (low background, sparse)
    X = np.random.negative_binomial(1, 0.5, size=(n_cells, n_genes)).astype(np.float32)
    X[X < 2] = 0

    # For each cell type, boost specific gene subsets to simulate signaling
    genes_per_type = n_genes // n_cell_types
    for ct_idx in range(n_cell_types):
        ct_mask = labels == cell_types[ct_idx]
        n_ct = ct_mask.sum()
        if n_ct == 0:
            continue

        # Boost a contiguous block of genes for this cell type (as ligands)
        start_lig = (ct_idx * genes_per_type) % n_genes
        lig_indices = [(start_lig + j) % n_genes for j in range(genes_per_type // 2)]

        # Boost a different block (as receptors on other cells)
        start_rec = (start_lig + n_genes // 3) % n_genes
        rec_indices = [(start_rec + j) % n_genes for j in range(genes_per_type // 2)]

        # Ligands on this cell type
        for li in lig_indices:
            X[ct_mask, li] += np.random.negative_binomial(5, 0.3, size=n_ct).astype(np.float32)

        # Receptors on other cell types (paracrine)
        other_mask = ~ct_mask
        n_other = other_mask.sum()
        if n_other > 0:
            boost_size = min(n_other // 2, n_ct * 2)
            if boost_size > 0:
                boost_idx = np.random.choice(np.where(other_mask)[0], size=boost_size, replace=False)
                for ri in rec_indices:
                    X[boost_idx, ri] += np.random.negative_binomial(4, 0.3, size=len(boost_idx)).astype(np.float32)

    import anndata
    adata = anndata.AnnData(
        X=X,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs["cell_type"] = pd.Categorical(labels)
    return adata


def generate_spatial_dataset(n_cells=50000, n_cell_types=14, n_genes=3000, seed=42):
    """Generate spatial dataset with clustered cell types."""
    adata = generate_structured_dataset(n_cells, n_cell_types, n_genes, seed)

    np.random.seed(seed + 1)
    coords = np.zeros((n_cells, 2))
    labels = adata.obs["cell_type"].values
    unique_types = sorted(adata.obs["cell_type"].unique())

    # Place each cell type in a spatial cluster
    cols = 4
    for i, ct in enumerate(unique_types):
        mask = labels == ct
        n = mask.sum()
        cx = (i % cols) * 250 + 125
        cy = (i // cols) * 250 + 125
        coords[mask, 0] = cx + np.random.randn(n) * 50
        coords[mask, 1] = cy + np.random.randn(n) * 50

    adata.obsm["spatial"] = coords
    return adata


def run_pipeline(adata, label, nboot=10, spatial=False):
    """Run the full pycellchat pipeline and report timing."""
    import pycellchat
    from pycellchat.modeling import compute_commun_prob, compute_commun_prob_pathway, aggregate_net, CommunProbParams
    from pycellchat.analysis import net_analysis_compute_centrality, identify_communication_patterns, compute_net_similarity

    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"{'='*60}")

    timings = {}

    with timer("Create CellChat") as t:
        kwargs = {"group_by": "cell_type"}
        if spatial:
            kwargs["datatype"] = "spatial"
            kwargs["coordinates"] = adata.obsm["spatial"]
        cc = pycellchat.CellChat(adata, **kwargs)
    timings["create"] = t.elapsed

    with timer("Load database") as t:
        cc.set_db("human")
    timings["db_load"] = t.elapsed

    with timer("Normalize") as t:
        cc.normalize()
    timings["normalize"] = t.elapsed

    with timer("Subset signaling genes") as t:
        cc.subset_data()
    timings["subset"] = t.elapsed
    n_sig = len(cc.cc["signaling_genes"])
    print(f"  Signaling genes: {n_sig}")

    with timer("Compute communication probability") as t:
        params = CommunProbParams(nboot=nboot, seed=42)
        compute_commun_prob(cc, params)
    timings["commun_prob"] = t.elapsed
    prob_shape = cc.cc["net"]["prob"].shape
    n_nonzero = (cc.cc["net"]["prob"] > 0).sum()
    print(f"  Prob shape: {prob_shape}")
    print(f"  Non-zero entries: {n_nonzero:,}")

    with timer("Pathway aggregation") as t:
        compute_commun_prob_pathway(cc)
    timings["pathway"] = t.elapsed
    n_pathways = len(cc.cc["netP"]["pathways"])
    print(f"  Significant pathways: {n_pathways}")

    with timer("Network aggregation") as t:
        aggregate_net(cc)
    timings["aggregate"] = t.elapsed

    with timer("Centrality metrics") as t:
        net_analysis_compute_centrality(cc)
    timings["centrality"] = t.elapsed

    with timer("Communication patterns (NMF)") as t:
        k = max(1, min(3, n_pathways)) if n_pathways > 0 else 0
        if k > 0:
            identify_communication_patterns(cc, pattern="outgoing", k=k)
        else:
            print("    Skipping NMF (no significant pathways)")
    timings["patterns"] = t.elapsed

    with timer("Network similarity") as t:
        if n_pathways > 1:
            compute_net_similarity(cc)
        else:
            print("    Skipping similarity (< 2 pathways)")
    timings["similarity"] = t.elapsed

    prob_mb = cc.cc["net"]["prob"].nbytes / 1024 / 1024
    pval_mb = cc.cc["net"]["pval"].nbytes / 1024 / 1024
    data_mb = cc.cc["data.signaling"].nbytes / 1024 / 1024
    print(f"\n  Memory:")
    print(f"    data.signaling: {data_mb:.1f} MB")
    print(f"    net.prob: {prob_mb:.1f} MB")
    print(f"    net.pval: {pval_mb:.1f} MB")
    print(f"    Total: {prob_mb + pval_mb + data_mb:.1f} MB")

    total_time = sum(timings.values())
    print(f"\n  Total pipeline time: {total_time:.2f}s")

    del cc
    gc.collect()
    return timings


def main():
    print("=" * 60)
    print("pycellchat Benchmark (3000 genes, 14 types, ~2000 signaling)")
    print("=" * 60)

    results = {}

    # 100K cells
    print("\n[1/3] Generating 100K cell dataset...")
    with timer("Generation"):
        adata_100k = generate_structured_dataset(100000, n_cell_types=14, n_genes=3000)
    print(f"  Shape: {adata_100k.shape}, Memory: {adata_100k.X.nbytes / 1024 / 1024:.0f} MB")
    results["100K"] = run_pipeline(adata_100k, "100K cells, 3000 genes, 14 types", nboot=10)
    del adata_100k; gc.collect()

    # 1M cells
    print("\n[2/3] Generating 1M cell dataset...")
    with timer("Generation"):
        adata_1m = generate_structured_dataset(1000000, n_cell_types=14, n_genes=3000)
    print(f"  Shape: {adata_1m.shape}, Memory: {adata_1m.X.nbytes / 1024 / 1024:.0f} MB")
    results["1M"] = run_pipeline(adata_1m, "1M cells, 3000 genes, 14 types", nboot=5)
    del adata_1m; gc.collect()

    # 50K spatial
    print("\n[3/3] Generating 50K spatial dataset...")
    with timer("Generation"):
        adata_spatial = generate_spatial_dataset(50000, n_cell_types=14, n_genes=3000)
    print(f"  Shape: {adata_spatial.shape}")
    results["50K_spatial"] = run_pipeline(adata_spatial, "50K cells, 3000 genes, 14 types (spatial)", nboot=10)
    del adata_spatial; gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<16} {'Total':>8} {'CommProb':>10} {'Normalize':>10} {'Subset':>8} {'Signaling':>10} {'Pathways':>10} {'Memory':>10}")
    print("-" * 82)
    for label, t in results.items():
        total = sum(t.values())
        print(f"{label:<16} {total:>7.1f}s {t.get('commun_prob',0):>9.1f}s {t.get('normalize',0):>9.1f}s {t.get('subset',0):>7.1f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
