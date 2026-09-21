import squidpy as sq
import numpy as np


def top_partner(zscore, cats, target):
    i = cats.index(target)
    row = zscore[i].copy()
    row[i] = -np.inf
    j = int(np.argmax(row))
    return cats[j], row[j]


if __name__ == "__main__":
    adata = sq.datasets.imc()
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    sq.gr.nhood_enrichment(adata, cluster_key="cell type", seed=0, n_jobs=1)

    zscore = adata.uns["cell type_nhood_enrichment"]["zscore"]
    cats = list(adata.obs["cell type"].cat.categories)

    # train
    print("train (T cells):", top_partner(zscore, cats, "T cells"))
    # test
    print(
        "test (proliferative tumor cell):",
        top_partner(zscore, cats, "proliferative tumor cell"),
    )
