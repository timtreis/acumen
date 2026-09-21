import squidpy as sq


def top_gene(adata):
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.spatial_autocorr(adata, mode="moran", seed=0, n_jobs=1)
    return adata.uns["moranI"].head(10)


def main():
    train_adata = sq.datasets.slideseqv2()
    print("TRAIN (slideseqv2):")
    print(top_gene(train_adata))

    test_adata = sq.datasets.seqfish()
    print("TEST (seqfish):")
    print(top_gene(test_adata))


if __name__ == "__main__":
    main()
