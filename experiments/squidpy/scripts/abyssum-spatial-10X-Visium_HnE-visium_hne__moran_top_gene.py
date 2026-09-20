import squidpy as sq


def top_moran_gene(adata):
    sq.gr.spatial_neighbors(adata)
    genes = adata[:, adata.var.highly_variable].var_names.values[:1000]
    sq.gr.spatial_autocorr(adata, mode="moran", genes=genes, n_perms=None, n_jobs=1, show_progress_bar=False)
    return adata.uns["moranI"].index[0]


# train: visium_hne data
adata_hne = sq.datasets.visium_hne_adata()
print("train answer:", top_moran_gene(adata_hne))

# test: slideseqv2 data
adata_slide = sq.datasets.slideseqv2()
print("test answer:", top_moran_gene(adata_slide))
