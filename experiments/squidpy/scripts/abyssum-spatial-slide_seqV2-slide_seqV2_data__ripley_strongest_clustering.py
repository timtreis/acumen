import squidpy as sq


def top_cluster(adata, cluster_key, **kwargs):
    res = sq.gr.ripley(adata, cluster_key=cluster_key, mode="L", seed=0, copy=True, **kwargs)
    df = res["L_stat"]
    top = df.loc[df["stats"].idxmax()]
    return top[cluster_key], top["stats"]


def main():
    train_adata = sq.datasets.slideseqv2()
    print("TRAIN (slideseqv2, cluster):", top_cluster(train_adata, "cluster", max_dist=500))

    test_adata = sq.datasets.seqfish()
    print("TEST (seqfish, celltype_mapped_refined):", top_cluster(test_adata, "celltype_mapped_refined"))


if __name__ == "__main__":
    main()
