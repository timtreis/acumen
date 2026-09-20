"""Confirms answers for the co-occurrence task (train + test)."""
import squidpy as sq
import pandas as pd

adata = sq.datasets.seqfish()
sq.gr.co_occurrence(adata, cluster_key="celltype_mapped_refined")
res = adata.uns["celltype_mapped_refined_co_occurrence"]
occ = res["occ"]  # (n_clusters, n_clusters, n_intervals)
cats = adata.obs["celltype_mapped_refined"].cat.categories.tolist()


def top_at_short_range(target):
    i = cats.index(target)
    scores = occ[i, :, 0]  # smallest distance interval
    s = pd.Series(scores, index=cats).drop(index=target)
    return s.sort_values(ascending=False)


for target in ["Lateral plate mesoderm", "Endothelium"]:
    s = top_at_short_range(target)
    print(target, "-> TRAIN/TEST answer:", s.index[0], round(s.iloc[0], 3))
    print(s.head(3))
