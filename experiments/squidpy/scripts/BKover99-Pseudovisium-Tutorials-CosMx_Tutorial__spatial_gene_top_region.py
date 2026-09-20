import numpy as np
import pandas as pd
import squidpy as sq

gene = "Plp1"

# train: visium H&E mouse brain
adata = sq.datasets.visium_hne_adata()
sq.pl.spatial_scatter(adata, color=gene, img=False, save=None)
expr = np.array(adata[:, gene].X.todense()).flatten()
df = pd.DataFrame({"expr": expr, "region": adata.obs["cluster"].values})
top = df.groupby("region")["expr"].mean().sort_values(ascending=False)
print("train (hne) top region for", gene, ":", top.index[0], top.head(3))

# test: visium fluorescent mouse brain
adata2 = sq.datasets.visium_fluo_adata()
sq.pl.spatial_scatter(adata2, color=gene, img=False, save=None)
expr2 = np.array(adata2[:, gene].X.todense()).flatten()
df2 = pd.DataFrame({"expr": expr2, "region": adata2.obs["cluster"].values})
top2 = df2.groupby("region")["expr"].mean().sort_values(ascending=False)
print("test (fluo) top region for", gene, ":", top2.index[0], top2.head(3))
