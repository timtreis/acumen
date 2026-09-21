import numpy as np
import pandas as pd
import squidpy as sq


def top_enrichment_partner(adata, cluster_key, target):
    zscore = np.array(adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"], dtype=float).copy()
    cats = list(adata.obs[cluster_key].cat.categories)
    np.fill_diagonal(zscore, -np.inf)
    df = pd.DataFrame(zscore, index=cats, columns=cats)
    col = df[target]
    return col.idxmax(), float(col.max())


if __name__ == "__main__":
    adata = sq.datasets.imc()
    adata.obs["cell type"] = adata.obs["cell type"].astype("category")
    key = "cell type"

    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=False, n_neighs=10)
    sq.gr.nhood_enrichment(adata, cluster_key=key, seed=0, show_progress_bar=False)

    train_partner, train_z = top_enrichment_partner(adata, key, "T cells")
    print("TRAIN (target='T cells'):", train_partner, round(train_z, 2))

    test_partner, test_z = top_enrichment_partner(adata, key, "macrophages")
    print("TEST (target='macrophages'):", test_partner, round(test_z, 2))
