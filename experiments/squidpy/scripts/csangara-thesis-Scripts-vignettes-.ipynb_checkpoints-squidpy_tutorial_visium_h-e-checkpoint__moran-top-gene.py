import squidpy as sq


def top_autocorrelated_gene(adata):
    sq.gr.spatial_neighbors(adata)
    res = sq.gr.spatial_autocorr(
        adata, mode="moran", n_perms=None, n_jobs=1, show_progress_bar=False, copy=True
    )
    return res.index[0], res.iloc[0]["I"]


# train: visium_hne data
adata_train = sq.datasets.visium_hne_adata()
gene_train, i_train = top_autocorrelated_gene(adata_train)
print("train:", gene_train, i_train)

# test: slideseqv2 data
adata_test = sq.datasets.slideseqv2()
gene_test, i_test = top_autocorrelated_gene(adata_test)
print("test:", gene_test, i_test)
