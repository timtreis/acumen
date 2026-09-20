import os
os.environ.setdefault("TMPDIR", "/tmp")

import numpy as np
import squidpy as sq


def most_segregated_pair(adata, cluster_key, spatial_key="spatial", coord_type=None):
    kwargs = {}
    if coord_type is not None:
        kwargs["coord_type"] = coord_type
    sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, **kwargs)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=0, n_jobs=1)

    zscore = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = list(adata.obs[cluster_key].cat.categories)
    n = len(cats)

    worst, worst_val = None, np.inf
    for i in range(n):
        for j in range(i + 1, n):
            val = (zscore[i, j] + zscore[j, i]) / 2
            if val < worst_val:
                worst_val = val
                worst = (cats[i], cats[j])
    return worst, worst_val


if __name__ == "__main__":
    # train: imc dataset, "cell type" labels
    adata_train = sq.datasets.imc()
    pair, val = most_segregated_pair(adata_train, "cell type")
    print("TRAIN most segregated pair:", pair, val)
    # -> ('apoptotic tumor cell', 'macrophages')

    # test: merfish dataset (multiple combined tissue sections), "Cell_class" labels
    adata_test = sq.datasets.merfish()
    pair, val = most_segregated_pair(
        adata_test, "Cell_class", spatial_key="spatial3d", coord_type="generic"
    )
    print("TEST most segregated pair:", pair, val)
    # -> ('Inhibitory', 'OD Mature 2')
