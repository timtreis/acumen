import squidpy as sq


def most_clustered(adata, cluster_key):
    sq.gr.ripley(adata, cluster_key=cluster_key, mode="L", seed=0)
    df = adata.uns[f"{cluster_key}_ripley_L"]["L_stat"]
    max_bin = df["bins"].max()
    sub = df[df["bins"] == max_bin].sort_values("stats", ascending=False)
    return sub.iloc[0][cluster_key]


# train: slideseqv2 dataset, "cluster" annotation
train_adata = sq.datasets.slideseqv2()
print("TRAIN ANSWER:", most_clustered(train_adata, "cluster"))

# test: imc dataset, "cell type" annotation
test_adata = sq.datasets.imc()
print("TEST ANSWER:", most_clustered(test_adata, "cell type"))
