"""Ground-truth script for task: dominant_cluster_expression (train + test)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import squidpy as sq


def top_cluster_for_gene(adata, gene):
    expr = adata[:, gene].X
    expr = expr.toarray().ravel() if sp.issparse(expr) else np.asarray(expr).ravel()
    means = pd.Series(expr, index=adata.obs_names).groupby(adata.obs["cluster"]).mean()
    return means.idxmax(), means.sort_values(ascending=False)


# --- train variant ---
adata_hne = sq.datasets.visium_hne_adata()
top, means = top_cluster_for_gene(adata_hne, "Sox8")
print("train: gene=Sox8, dataset=visium_hne")
print(means.head(3))
print("answer:", top)

# --- test variant ---
adata_fluo = sq.datasets.visium_fluo_adata()
top, means = top_cluster_for_gene(adata_fluo, "Neurod6")
print("\ntest: gene=Neurod6, dataset=visium_fluo")
print(means.head(3))
print("answer:", top)
