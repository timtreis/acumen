import squidpy as sq


def main():
    for name, loader in [
        ("visium_hne", sq.datasets.visium_hne_adata),
        ("visium_fluo", sq.datasets.visium_fluo_adata),
    ]:
        adata = loader()
        sq.gr.spatial_neighbors(adata)
        genes = adata[:, adata.var.highly_variable].var_names.values
        sq.gr.spatial_autocorr(adata, genes=genes, mode="moran", n_jobs=1)
        top = adata.uns["moranI"].head(3)
        print(name)
        print(top)
        print()

    # train answer (visium_hne): Nrgn
    # test answer (visium_fluo): Ttr


if __name__ == "__main__":
    main()
