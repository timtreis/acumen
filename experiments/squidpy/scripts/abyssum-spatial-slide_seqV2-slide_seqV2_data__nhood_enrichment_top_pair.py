import squidpy as sq


def top_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=0, n_jobs=1)
    zscore = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    n = len(cats)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            z = zscore[i, j]
            if best is None or z > best[0]:
                best = (z, cats[i], cats[j])
    return best


def main():
    train_adata = sq.datasets.slideseqv2()
    print("TRAIN (slideseqv2, cluster):", top_pair(train_adata, "cluster"))

    test_adata = sq.datasets.seqfish()
    print("TEST (seqfish, celltype_mapped_refined):", top_pair(test_adata, "celltype_mapped_refined"))


if __name__ == "__main__":
    main()
