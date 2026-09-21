import warnings
warnings.filterwarnings("ignore")
import squidpy as sq

def top_moran_gene(loader):
    adata = loader()
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names.tolist(), n_perms=None, n_jobs=1)
    moran = adata.uns["moranI"].sort_values("I", ascending=False)
    return moran.index[0], moran["I"].iloc[0]

if __name__ == "__main__":
    train_gene, train_i = top_moran_gene(sq.datasets.visium_hne_adata)
    print("train (visium_hne):", train_gene, train_i)

    test_gene, test_i = top_moran_gene(sq.datasets.visium_fluo_adata)
    print("test (visium_fluo):", test_gene, test_i)
