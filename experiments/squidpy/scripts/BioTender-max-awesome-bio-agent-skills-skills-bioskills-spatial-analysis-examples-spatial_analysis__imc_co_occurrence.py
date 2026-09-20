import squidpy as sq
import numpy as np

adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type")

occ = adata.uns["cell type_co_occurrence"]["occ"]
cats = list(adata.obs["cell type"].cat.categories)


def top_partner(target):
    i = cats.index(target)
    first = occ[i, :, 0].copy()
    first[i] = -np.inf
    j = int(np.argmax(first))
    return cats[j], first[j]


# train
print("train (vimentin hi stromal cell):", top_partner("vimentin hi stromal cell"))
# test
print("test (CK+ HR+ tumor cell):", top_partner("CK+ HR+ tumor cell"))
