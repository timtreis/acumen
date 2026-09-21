import os
os.environ.setdefault("TMPDIR", "/tmp")

import numpy as np
import squidpy as sq


def top_enriched_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=0, n_jobs=1)

    zscore = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = list(adata.obs[cluster_key].cat.categories)
    n = len(cats)

    best, best_val = None, -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            val = (zscore[i, j] + zscore[j, i]) / 2
            if val > best_val:
                best_val = val
                best = (cats[i], cats[j])
    return best, best_val


if __name__ == "__main__":
    # train: imc dataset, "cell type" labels
    adata_train = sq.datasets.imc()
    pair, val = top_enriched_pair(adata_train, "cell type")
    print("TRAIN top enriched pair:", pair, val)
    # -> ('T cells', 'endothelial')

    # test: mibitof dataset, "Cluster" labels
    adata_test = sq.datasets.mibitof()
    pair, val = top_enriched_pair(adata_test, "Cluster")
    print("TEST top enriched pair:", pair, val)
    # -> ('Tcell_CD4', 'Tcell_CD8')
