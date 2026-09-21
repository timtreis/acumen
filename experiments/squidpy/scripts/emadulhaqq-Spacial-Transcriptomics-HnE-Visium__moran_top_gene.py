import squidpy as sq


def train():
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", n_jobs=1)
    top_gene = adata.uns["moranI"].index[0]
    print("TRAIN top spatially autocorrelated gene (visium_hne):", top_gene)


def test():
    adata = sq.datasets.slideseqv2()
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran", n_jobs=1)
    top_gene = adata.uns["moranI"].index[0]
    print("TEST top spatially autocorrelated gene (slideseqv2):", top_gene)


if __name__ == "__main__":
    train()
    test()
