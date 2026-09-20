import numpy as np
import pandas as pd
import squidpy as sq

adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cluster", show_progress_bar=False)

cats = adata.obs["cluster"].cat.categories.tolist()
z = np.array(adata.uns["cluster_nhood_enrichment"]["zscore"], dtype=float, copy=True)
df = pd.DataFrame(z, index=cats, columns=cats)
for c in cats:
    df.loc[c, c] = -np.inf

# train: strongest spatial-neighbor partner of Hippocampus
print("train target: Hippocampus")
print(df["Hippocampus"].sort_values(ascending=False).head(3))

# test: strongest spatial-neighbor partner of Pyramidal_layer_dentate_gyrus
print("\ntest target: Pyramidal_layer_dentate_gyrus")
print(df["Pyramidal_layer_dentate_gyrus"].sort_values(ascending=False).head(3))
