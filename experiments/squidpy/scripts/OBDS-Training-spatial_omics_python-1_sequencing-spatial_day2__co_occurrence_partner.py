import squidpy as sq
import pandas as pd

adata = sq.datasets.visium_hne_adata()
sq.gr.co_occurrence(adata, cluster_key="cluster")

occ = adata.uns["cluster_co_occurrence"]["occ"]
cats = adata.obs["cluster"].cat.categories.tolist()
idx = {c: i for i, c in enumerate(cats)}

for query in ["Pyramidal_layer", "Striatum"]:
    i = idx[query]
    s = pd.Series(occ[i, :, 0], index=cats).drop(query).sort_values(ascending=False)
    print(query, "-> at the shortest distance interval, highest co-occurrence with:", s.index[0], s.iloc[0], "| runner-up:", s.index[1], s.iloc[1])
