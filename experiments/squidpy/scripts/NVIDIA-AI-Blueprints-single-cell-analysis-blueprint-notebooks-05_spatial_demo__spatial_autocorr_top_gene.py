import squidpy as sq


def top_autocorrelated_gene(dataset_fn):
    adata = dataset_fn()
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", n_jobs=-1)
    top = adata.uns["moranI"].sort_values("I", ascending=False).index[0]
    return top, adata.uns["moranI"].head(5)


# train: visium_hne dataset
top_hne, table_hne = top_autocorrelated_gene(sq.datasets.visium_hne_adata)
print("train (visium_hne) top gene:", top_hne)
print(table_hne)

# test: visium_fluo dataset
top_fluo, table_fluo = top_autocorrelated_gene(sq.datasets.visium_fluo_adata)
print("test (visium_fluo) top gene:", top_fluo)
print(table_fluo)
