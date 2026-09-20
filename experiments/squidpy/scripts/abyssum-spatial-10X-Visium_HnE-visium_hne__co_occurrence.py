import pandas as pd
import squidpy as sq


def closest_range_partner(adata, cluster_key, target_cluster):
    occ, interval = sq.gr.co_occurrence(adata, cluster_key=cluster_key, copy=True)
    cats = adata.obs[cluster_key].cat.categories.tolist()
    idx = cats.index(target_cluster)
    df = pd.DataFrame(occ[idx], index=cats)
    sub = df.drop(target_cluster)
    closest_interval = sub.iloc[:, 0]
    return closest_interval.idxmax(), closest_interval.max()


# train: visium_hne data, target region = Fiber_tract
adata_hne = sq.datasets.visium_hne_adata()
train_answer = closest_range_partner(adata_hne, "cluster", "Fiber_tract")
print("train answer:", train_answer)

# test: visium_fluo data, target region = Hippocampus
adata_fluo = sq.datasets.visium_fluo_adata()
test_answer = closest_range_partner(adata_fluo, "cluster", "Hippocampus")
print("test answer:", test_answer)
