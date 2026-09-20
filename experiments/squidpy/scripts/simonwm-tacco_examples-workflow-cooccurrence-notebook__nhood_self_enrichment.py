import squidpy as sq


def main():
    # --- train: imc dataset ---
    adata = sq.datasets.imc()
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cell type", n_jobs=1, seed=0)
    z = adata.uns["cell type_nhood_enrichment"]["zscore"]
    cats = list(adata.obs["cell type"].cat.categories)
    diag = [(cats[i], z[i, i]) for i in range(len(cats))]
    diag.sort(key=lambda x: -x[1])
    print("train answer:", diag[0])

    # --- test: seqfish dataset ---
    adata2 = sq.datasets.seqfish()
    sq.gr.spatial_neighbors(adata2)
    sq.gr.nhood_enrichment(adata2, cluster_key="celltype_mapped_refined", n_jobs=1, seed=0)
    z2 = adata2.uns["celltype_mapped_refined_nhood_enrichment"]["zscore"]
    cats2 = list(adata2.obs["celltype_mapped_refined"].cat.categories)
    diag2 = [(cats2[i], z2[i, i]) for i in range(len(cats2))]
    diag2.sort(key=lambda x: -x[1])
    print("test answer:", diag2[0])


if __name__ == "__main__":
    main()
