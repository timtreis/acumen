import squidpy as sq

# train: visium H&E mouse brain
adata = sq.datasets.visium_hne_adata()
sq.pl.spatial_scatter(adata, color="total_counts", img=False, save=None)
print("train (hne) max total_counts:", adata.obs["total_counts"].max())

# test: visium fluorescent mouse brain
adata2 = sq.datasets.visium_fluo_adata()
sq.pl.spatial_scatter(adata2, color="total_counts", img=False, save=None)
print("test (fluo) max total_counts:", adata2.obs["total_counts"].max())
