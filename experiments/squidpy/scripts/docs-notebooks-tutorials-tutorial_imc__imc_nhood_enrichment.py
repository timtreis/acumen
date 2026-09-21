"""
Ground truth for task `imc_nhood_enrichment`.

Builds the spatial neighbor graph for the imc dataset and runs a
permutation-based neighborhood enrichment test on the "cell type"
annotation. For a given target cell type, reports which other cell type
has the highest enrichment z-score (i.e. is found next to the target more
often than expected by chance).
"""

import pandas as pd
import squidpy as sq

if __name__ == "__main__":
    adata = sq.datasets.imc()

    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cell type", n_jobs=1, seed=0)

    cats = list(adata.obs["cell type"].cat.categories)
    zscore = adata.uns["cell type_nhood_enrichment"]["zscore"]
    df = pd.DataFrame(zscore, index=cats, columns=cats)

    for target in ["T cells", "macrophages"]:
        row = df.loc[target].drop(target).sort_values(ascending=False)
        print(f"{target} -> top enriched neighbor: {row.index[0]} (z={row.iloc[0]:.2f}), "
              f"2nd: {row.index[1]} (z={row.iloc[1]:.2f})")

    # train answer (target = "T cells"): endothelial
    # test answer (target = "macrophages"): T cells
