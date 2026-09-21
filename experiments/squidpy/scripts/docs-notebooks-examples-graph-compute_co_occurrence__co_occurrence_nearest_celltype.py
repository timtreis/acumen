import numpy as np
import squidpy as sq

adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type")

res = adata.uns["cell type_co_occurrence"]
occ = res["occ"]  # shape (n_clusters, n_clusters, n_intervals)
cats = adata.obs["cell type"].cat.categories.tolist()


def most_co_occurring_at_shortest_distance(target):
    idx = cats.index(target)
    vals = occ[idx, :, 0].copy()
    vals[idx] = -np.inf
    best_idx = int(np.argmax(vals))
    return cats[best_idx], vals[best_idx]


# train variant
train_target = "basal CK tumor cell"
print("train:", train_target, "->", most_co_occurring_at_shortest_distance(train_target))

# test variant
test_target = "T cells"
print("test:", test_target, "->", most_co_occurring_at_shortest_distance(test_target))
