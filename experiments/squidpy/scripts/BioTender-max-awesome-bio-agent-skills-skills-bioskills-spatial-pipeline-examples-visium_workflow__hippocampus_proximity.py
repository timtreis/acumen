import numpy as np
import squidpy as sq


def closest_neighbor_of_hippocampus(adata):
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    sq.gr.co_occurrence(adata, cluster_key="cluster")
    occ = adata.uns["cluster_co_occurrence"]["occ"]
    cats = adata.obs["cluster"].cat.categories.tolist()
    hidx = cats.index("Hippocampus")
    first = occ[hidx, :, 0].copy()
    first[hidx] = -np.inf
    j = np.argmax(first)
    return cats[j], first[j]


# train: visium_hne
adata_train = sq.datasets.visium_hne_adata()
print("train:", closest_neighbor_of_hippocampus(adata_train))

# test: visium_fluo
adata_test = sq.datasets.visium_fluo_adata()
print("test:", closest_neighbor_of_hippocampus(adata_test))
