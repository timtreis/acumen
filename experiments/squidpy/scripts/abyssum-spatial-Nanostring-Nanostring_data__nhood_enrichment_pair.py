import os

os.environ["TMPDIR"] = "/tmp"

import numpy as np
import pandas as pd

import squidpy as sq


def top_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1, seed=0)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories
    df = pd.DataFrame(z, index=cats, columns=cats)
    arr = df.values.copy()
    np.fill_diagonal(arr, -np.inf)
    r, c = np.unravel_index(np.argmax(arr), arr.shape)
    return cats[r], cats[c], arr[r, c]


if __name__ == "__main__":
    # train: seqfish dataset, cluster key "celltype_mapped_refined"
    adata = sq.datasets.seqfish()
    print("train answer:", top_pair(adata, "celltype_mapped_refined"))

    # test: imc dataset, cluster key "cell type"
    adata2 = sq.datasets.imc()
    print("test answer:", top_pair(adata2, "cell type"))
