import squidpy as sq
import numpy as np


def top_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="generic")
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key)

    im = np.array(adata.uns[f"{cluster_key}_interactions"], dtype=float).copy()
    cats = list(adata.obs[cluster_key].cat.categories)

    mat = im.copy()
    np.fill_diagonal(mat, -1)
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    return cats[idx[0]], cats[idx[1]], mat[idx]


# train: imc data
adata_train = sq.datasets.imc()
print("TRAIN answer:", top_pair(adata_train, "cell type"))

# test: seqfish data
adata_test = sq.datasets.seqfish()
print("TEST answer:", top_pair(adata_test, "celltype_mapped_refined"))
