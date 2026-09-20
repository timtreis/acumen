import squidpy as sq


def top_spatially_variable_gene(adata):
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=1, show_progress_bar=False, seed=0)
    svg = adata.uns["moranI"].sort_values("I", ascending=False)
    return svg.index[0], svg["I"].iloc[0]


# train: visium_hne
adata_train = sq.datasets.visium_hne_adata()
print("train:", top_spatially_variable_gene(adata_train))

# test: visium_fluo
adata_test = sq.datasets.visium_fluo_adata()
print("test:", top_spatially_variable_gene(adata_test))
