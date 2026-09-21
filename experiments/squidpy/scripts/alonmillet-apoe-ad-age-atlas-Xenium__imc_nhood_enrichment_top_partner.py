import squidpy as sq
import pandas as pd

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
sq.gr.nhood_enrichment(adata, cluster_key="cell type", seed=0, n_jobs=1, show_progress_bar=False)

zscore = adata.uns["cell type_nhood_enrichment"]["zscore"]
cats = adata.obs["cell type"].cat.categories.tolist()
df = pd.DataFrame(zscore, index=cats, columns=cats)


def top_partner(target):
    ranked = df[target].drop(target).sort_values(ascending=False)
    return ranked.index[0], ranked.iloc[0]


# train
print("train (target=macrophages):", top_partner("macrophages"))
# test
print("test (target=small elongated stromal cell):", top_partner("small elongated stromal cell"))
