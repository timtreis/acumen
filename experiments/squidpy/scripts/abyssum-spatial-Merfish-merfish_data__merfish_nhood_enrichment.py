import squidpy as sq
import pandas as pd

adata = sq.datasets.merfish()
sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial3d")
sq.gr.nhood_enrichment(adata, cluster_key="Cell_class", seed=0, n_jobs=1, show_progress_bar=False)

zscore = adata.uns["Cell_class_nhood_enrichment"]["zscore"]
cats = adata.obs["Cell_class"].cat.categories.tolist()
df = pd.DataFrame(zscore, index=cats, columns=cats)

for target in ["Pericytes", "OD Mature 1"]:
    row = df.loc[target].drop(target)
    print(target, "-> top enriched neighbor:", row.idxmax(), round(row.max(), 2))

# train answer: Pericytes -> Endothelial 2
# test answer: OD Mature 1 -> OD Mature 2
