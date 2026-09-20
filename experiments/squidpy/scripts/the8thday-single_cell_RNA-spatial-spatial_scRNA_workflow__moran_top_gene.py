import squidpy as sq

# TRAIN: visium_hne_adata
adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names, n_jobs=1)
moran = adata.uns["moranI"]
top_gene_train = moran.sort_values("I", ascending=False).index[0]
print("TRAIN top spatially variable gene:", top_gene_train)

# TEST: visium_fluo_adata
adata2 = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata2)
sq.gr.spatial_autocorr(adata2, mode="moran", genes=adata2.var_names, n_jobs=1)
moran2 = adata2.uns["moranI"]
top_gene_test = moran2.sort_values("I", ascending=False).index[0]
print("TEST top spatially variable gene:", top_gene_test)
