import squidpy as sq


def top_moran_gene(adata):
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=None, n_jobs=1, show_progress_bar=False)
    return adata.uns["moranI"].index[0]


# TRAIN: visium_hne_adata dataset (mouse brain, H&E)
adata_train = sq.datasets.visium_hne_adata()
print("TRAIN answer:", top_moran_gene(adata_train))

# TEST: visium_fluo_adata dataset (mouse brain, fluorescence)
adata_test = sq.datasets.visium_fluo_adata()
print("TEST answer:", top_moran_gene(adata_test))
