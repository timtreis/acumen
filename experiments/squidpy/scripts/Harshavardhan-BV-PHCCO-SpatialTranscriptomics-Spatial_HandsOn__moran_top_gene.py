import squidpy as sq
import scanpy as sc

# --- train: H&E mouse brain Visium dataset ---
adata = sq.datasets.visium_hne_adata()
sc.pp.highly_variable_genes(adata)
sq.gr.spatial_neighbors(adata)
genes = adata[:, adata.var.highly_variable].var_names.values
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    genes=genes,
    n_perms=None,
    n_jobs=1,
    show_progress_bar=False,
)
print("train (visium_hne) top gene by Moran's I:", adata.uns["moranI"].index[0])
print(adata.uns["moranI"].head(5))

# --- test: fluorescence mouse brain Visium dataset ---
adata2 = sq.datasets.visium_fluo_adata()
sc.pp.highly_variable_genes(adata2)
sq.gr.spatial_neighbors(adata2)
genes2 = adata2[:, adata2.var.highly_variable].var_names.values
sq.gr.spatial_autocorr(
    adata2,
    mode="moran",
    genes=genes2,
    n_perms=None,
    n_jobs=1,
    show_progress_bar=False,
)
print("test (visium_fluo) top gene by Moran's I:", adata2.uns["moranI"].index[0])
print(adata2.uns["moranI"].head(5))
