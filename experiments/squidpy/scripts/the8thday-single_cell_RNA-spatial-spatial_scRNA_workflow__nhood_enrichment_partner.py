import numpy as np
import pandas as pd
import squidpy as sq

# TRAIN: visium_hne_adata, which domain is most enriched next to Hippocampus?
adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, show_progress_bar=False)
z = np.asarray(adata.uns["cluster_nhood_enrichment"]["zscore"], dtype=float)
cats = adata.obs["cluster"].cat.categories.tolist()
df = pd.DataFrame(z, index=cats, columns=cats)
row = df.loc["Hippocampus"].drop("Hippocampus").sort_values(ascending=False)
print("TRAIN most enriched partner of Hippocampus:", row.index[0])
print(row.head())

# TEST: visium_fluo_adata, which domain is most enriched next to Dentate_gyrus?
adata2 = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata2)
sq.gr.nhood_enrichment(adata2, cluster_key="cluster", seed=0, show_progress_bar=False)
z2 = np.asarray(adata2.uns["cluster_nhood_enrichment"]["zscore"], dtype=float)
cats2 = adata2.obs["cluster"].cat.categories.tolist()
df2 = pd.DataFrame(z2, index=cats2, columns=cats2)
row2 = df2.loc["Dentate_gyrus"].drop("Dentate_gyrus").sort_values(ascending=False)
print("TEST most enriched partner of Dentate_gyrus:", row2.index[0])
print(row2.head())
