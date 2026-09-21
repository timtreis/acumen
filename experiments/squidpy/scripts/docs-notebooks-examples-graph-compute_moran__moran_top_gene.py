import squidpy as sq

# ---- train: visium_hne data ----
adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(adata, mode="moran", genes=None, n_perms=None, n_jobs=1)
top_train = adata.uns["moranI"].index[0]
print("train top gene (visium_hne):", top_train)

# ---- test: visium_fluo data ----
adata2 = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata2)
sq.gr.spatial_autocorr(adata2, mode="moran", genes=None, n_perms=None, n_jobs=1)
top_test = adata2.uns["moranI"].index[0]
print("test top gene (visium_fluo):", top_test)
