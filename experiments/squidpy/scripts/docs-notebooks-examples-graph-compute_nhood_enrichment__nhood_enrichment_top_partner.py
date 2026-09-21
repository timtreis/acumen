import pandas as pd
import squidpy as sq

adata = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, show_progress_bar=False)

cats = adata.obs["cluster"].cat.categories.tolist()
zscore = adata.uns["cluster_nhood_enrichment"]["zscore"]
df = pd.DataFrame(zscore, index=cats, columns=cats)

for target in ["Hippocampus", "Fiber_tracts"]:
    row = df.loc[target].drop(target)
    print(target, "->", row.idxmax(), round(row.max(), 2))
