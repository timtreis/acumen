import os

os.environ["TMPDIR"] = "/tmp"

import squidpy as sq

adata_full = sq.datasets.visium_hne_adata()


def top_spatially_variable_gene(cluster):
    adata = adata_full[adata_full.obs["cluster"] == cluster].copy()
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", n_jobs=1, show_progress_bar=False)
    top = adata.uns["moranI"].index[0]
    return top, adata.uns["moranI"]["I"].iloc[0]


train_answer, train_val = top_spatially_variable_gene("Hippocampus")
test_answer, test_val = top_spatially_variable_gene("Cortex_1")

print("train (Hippocampus) ->", train_answer, train_val)
print("test (Cortex_1) ->", test_answer, test_val)
