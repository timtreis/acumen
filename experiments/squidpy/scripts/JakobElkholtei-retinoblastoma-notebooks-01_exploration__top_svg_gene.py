import scanpy as sc
import squidpy as sq


def top_svg(adata):
    sq.gr.spatial_neighbors(adata)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, subset=True)
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=None)
    morans = adata.uns["moranI"].sort_values("I", ascending=False)
    return morans.index[0]


# train: visium_hne_adata
adata_train = sq.datasets.visium_hne_adata()
print("train (visium_hne_adata) top SVG:", top_svg(adata_train))

# test: visium_fluo_adata
adata_test = sq.datasets.visium_fluo_adata()
print("test (visium_fluo_adata) top SVG:", top_svg(adata_test))
