import squidpy as sq
import pandas as pd


def top_enriched_partner(adata, target):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cell type", n_jobs=1, show_progress_bar=False)
    z = adata.uns["cell type_nhood_enrichment"]["zscore"]
    cats = adata.obs["cell type"].cat.categories.tolist()
    df = pd.DataFrame(z, index=cats, columns=cats)
    row = df.loc[target].drop(target)
    return row.idxmax(), row.max()


# train
adata = sq.datasets.imc()
partner, z = top_enriched_partner(adata, "T cells")
print("train (T cells):", partner, z)

# test
adata = sq.datasets.imc()
partner, z = top_enriched_partner(adata, "macrophages")
print("test (macrophages):", partner, z)
