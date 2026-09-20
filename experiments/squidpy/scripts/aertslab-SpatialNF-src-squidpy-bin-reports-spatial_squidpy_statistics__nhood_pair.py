"""Task: nhood_pair
Find the pair of (different) cell types with the strongest spatial co-localization
(highest neighborhood-enrichment z-score, off-diagonal) in a bundled dataset.
"""
import numpy as np
import pandas as pd
import squidpy as sq


def top_offdiag_pair(adata, key, seed=0, library_key=None):
    if library_key is not None:
        sq.gr.spatial_neighbors(adata, library_key=library_key)
    else:
        sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=key, seed=seed)
    cats = adata.obs[key].cat.categories.tolist()
    zscore = np.array(adata.uns[f"{key}_nhood_enrichment"]["zscore"], dtype=float)
    off = zscore.copy()
    np.fill_diagonal(off, -np.inf)
    idx = np.unravel_index(np.argmax(off), off.shape)
    return cats[idx[0]], cats[idx[1]], off.max()


if __name__ == "__main__":
    # --- train: imc dataset ---
    adata_imc = sq.datasets.imc()
    pair_imc = top_offdiag_pair(adata_imc, "cell type", seed=0)
    print("TRAIN (imc):", pair_imc)
    # -> ('endothelial', 'T cells', ~19.3)

    # --- test: mibitof dataset ---
    adata_mibi = sq.datasets.mibitof()
    pair_mibi = top_offdiag_pair(adata_mibi, "Cluster", seed=0, library_key="library_id")
    print("TEST (mibitof):", pair_mibi)
    # -> ('Tcell_CD8', 'Tcell_CD4', ~7.5)
