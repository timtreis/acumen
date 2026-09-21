import squidpy as sq
import pandas as pd

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", seed=0, n_jobs=1, show_progress_bar=False)
z = adata.uns["cell type_nhood_enrichment"]["zscore"]
cats = list(adata.obs["cell type"].cat.categories)
df = pd.DataFrame(z, index=cats, columns=cats)

train_target = "T cells"
test_target = "small elongated stromal cell"

print("train answer:", df.loc[train_target].drop(train_target).idxmax())
print("test answer:", df.loc[test_target].drop(test_target).idxmax())
