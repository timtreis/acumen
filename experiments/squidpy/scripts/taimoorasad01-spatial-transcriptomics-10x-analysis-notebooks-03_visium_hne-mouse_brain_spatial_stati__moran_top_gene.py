import squidpy as sq

# train: visium_hne_adata
adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=1, seed=0, show_progress_bar=False)
print("TRAIN top spatially autocorrelated gene:", adata.uns["moranI"].index[0])
print(adata.uns["moranI"].head(10))

# test: seqfish (mouse embryo seqFISH)
adata2 = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata2)
sq.gr.spatial_autocorr(adata2, mode="moran", n_perms=100, n_jobs=1, seed=0, show_progress_bar=False)
print("TEST top spatially autocorrelated gene:", adata2.uns["moranI"].index[0])
print(adata2.uns["moranI"].head(10))
