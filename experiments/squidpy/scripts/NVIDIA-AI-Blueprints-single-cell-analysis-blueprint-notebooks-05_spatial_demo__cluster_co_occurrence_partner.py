import numpy as np
import squidpy as sq


def top_co_occurring_partner(dataset_fn, target):
    adata = dataset_fn()
    occ, interval = sq.gr.co_occurrence(adata, cluster_key="leiden", copy=True)
    categories = adata.obs["leiden"].cat.categories.tolist()
    ti = categories.index(target)
    first_bin = occ[ti, :, 0]
    order = np.argsort(first_bin)[::-1]
    ranked = [(categories[idx], first_bin[idx]) for idx in order]
    best_other = [c for c, _ in ranked if c != target][0]
    return best_other, ranked[:5]


# train: visium_hne dataset, target cluster "0"
best_hne, ranked_hne = top_co_occurring_partner(sq.datasets.visium_hne_adata, "0")
print("train (visium_hne, cluster 0) best partner:", best_hne)
print(ranked_hne)

# test: visium_fluo dataset, target cluster "3"
best_fluo, ranked_fluo = top_co_occurring_partner(sq.datasets.visium_fluo_adata, "3")
print("test (visium_fluo, cluster 3) best partner:", best_fluo)
print(ranked_fluo)
