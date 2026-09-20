import squidpy as sq
import pandas as pd


def top_partner(adata, target):
    sq.gr.spatial_neighbors(adata)
    sq.gr.interaction_matrix(adata, cluster_key="cell type")
    im = adata.uns["cell type_interactions"]
    cats = adata.obs["cell type"].cat.categories.tolist()
    df = pd.DataFrame(im, index=cats, columns=cats)
    row = df.loc[target].drop(target)
    return row.idxmax(), int(row.max())


# train
adata = sq.datasets.imc()
partner, count = top_partner(adata, "endothelial")
print("train (endothelial):", partner, count)

# test
adata = sq.datasets.imc()
partner, count = top_partner(adata, "macrophages")
print("test (macrophages):", partner, count)
