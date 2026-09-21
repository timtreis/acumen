"""
Ground truth for task `imc_co_occurrence`.

Computes the distance-dependent co-occurrence probability ratio between
cell types in the imc dataset. For a given target cell type, reports
which other cell type has the highest co-occurrence score with the target
at the shortest distance bin examined.
"""

import squidpy as sq

adata = sq.datasets.imc()

sq.gr.co_occurrence(adata, cluster_key="cell type")

res = adata.uns["cell type_co_occurrence"]
occ = res["occ"]  # shape (n_clusters, n_clusters, n_intervals)
cats = list(adata.obs["cell type"].cat.categories)

for target in ["p53+ EGFR+ tumor cell", "vimentin hi stromal cell"]:
    ti = cats.index(target)
    row0 = occ[ti, :, 0]
    ordered = sorted(zip(cats, row0), key=lambda x: -x[1])
    ordered = [(c, v) for c, v in ordered if c != target]
    print(f"{target} -> top co-occurring neighbor at shortest distance: "
          f"{ordered[0][0]} ({ordered[0][1]:.3f}), 2nd: {ordered[1][0]} ({ordered[1][1]:.3f})")

# train answer (target = "p53+ EGFR+ tumor cell"): proliferative tumor cell
# test answer (target = "vimentin hi stromal cell"): endothelial
