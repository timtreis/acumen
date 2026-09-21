import numpy as np
import pandas as pd
import squidpy as sq


def top_self_enrichment(adata, cluster_key, n_neighs=6):
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=n_neighs)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=0, show_progress_bar=False, n_jobs=1)
    zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore']
    cats = adata.obs[cluster_key].cat.categories.tolist()
    diag = np.diag(zscore)
    order = np.argsort(diag)[::-1]
    return [(cats[i], diag[i]) for i in order]


if __name__ == '__main__':
    # train: mibitof (MIBI-TOF breast cancer tissue, field of view "point16")
    mibi = sq.datasets.mibitof()
    sub = mibi[mibi.obs['library_id'] == 'point16'].copy()
    train_ranked = top_self_enrichment(sub, 'Cluster')
    print(train_ranked[:5])
    print("TRAIN TOP SELF-ENRICHED TYPE:", train_ranked[0][0])

    # test: imc (imaging mass cytometry breast cancer tissue)
    imc = sq.datasets.imc()
    test_ranked = top_self_enrichment(imc, 'cell type')
    print(test_ranked[:5])
    print("TEST TOP SELF-ENRICHED TYPE:", test_ranked[0][0])
