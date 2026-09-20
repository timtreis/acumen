"""Confirmation script for task 'nhood_self_enrichment' (train + test variants).

Goal: on the imc bundled dataset (obs['cell type'] holds cell-type labels),
build a spatial neighbor graph and run neighborhood enrichment to rank cell
types by how strongly they cluster with their own type in space (diagonal
of the enrichment z-score matrix).

train answer = cell type with the strongest self-clustering (highest
               self-vs-self z-score)
test answer  = cell type with the weakest self-clustering (lowest
               self-vs-self z-score)

Both extremes are stable across choices of neighbor-graph construction
(n_neighs=3/6/15/30/50, radius-based, Delaunay) -- checked separately.
"""
import numpy as np
import pandas as pd
import squidpy as sq

adata = sq.datasets.imc()

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=15)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", seed=0, n_jobs=1, show_progress_bar=False)

cats = adata.obs["cell type"].cat.categories.tolist()
z = adata.uns["cell type_nhood_enrichment"]["zscore"]
df = pd.DataFrame(z, index=cats, columns=cats)

diag = pd.Series(np.diag(df.values), index=cats).sort_values(ascending=False)

print("train answer (strongest self-clustering):", diag.index[0])
print("test answer (weakest self-clustering):", diag.index[-1])
print(diag)
