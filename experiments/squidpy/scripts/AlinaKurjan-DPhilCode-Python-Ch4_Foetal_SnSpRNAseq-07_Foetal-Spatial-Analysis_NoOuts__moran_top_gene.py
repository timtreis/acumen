import squidpy as sq


def top_gene(adata):
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=None, n_jobs=1)
    return adata.uns["moranI"].index[0], adata.uns["moranI"]["I"].iloc[0]


# train: visium_hne data
adata_hne = sq.datasets.visium_hne_adata()
print("train (visium_hne):", top_gene(adata_hne))

# test: visium_fluo data
adata_fluo = sq.datasets.visium_fluo_adata()
print("test (visium_fluo):", top_gene(adata_fluo))
