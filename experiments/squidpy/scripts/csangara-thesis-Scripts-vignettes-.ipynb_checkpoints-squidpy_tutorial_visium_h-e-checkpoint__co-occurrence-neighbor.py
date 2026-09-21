import numpy as np
import squidpy as sq


def top_co_occurring_other(adata, cluster_key, focus_cluster):
    occ, interval = sq.gr.co_occurrence(adata, cluster_key=cluster_key, copy=True)
    cats = list(adata.obs[cluster_key].cat.categories)
    idx = cats.index(focus_cluster)
    first_bin = occ[idx, :, 0].copy()
    first_bin[idx] = -np.inf  # exclude self
    best = np.argmax(first_bin)
    return cats[best], first_bin[best]


# train: visium_hne data, Hippocampus region
adata_train = sq.datasets.visium_hne_adata()
other_train, val_train = top_co_occurring_other(adata_train, "cluster", "Hippocampus")
print("train:", other_train, val_train)

# test: imc data, endothelial cells
adata_test = sq.datasets.imc()
other_test, val_test = top_co_occurring_other(adata_test, "cell type", "endothelial")
print("test:", other_test, val_test)
