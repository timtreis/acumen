# Confirms the answer for task "ripley_l_top_cluster" (train + test).
import squidpy as sq


def top_clustered_cluster(adata, cluster_key):
    res = sq.gr.ripley(adata, cluster_key=cluster_key, mode="L", copy=True, n_simulations=10)
    df = res["L_stat"]
    last_bin = df["bins"].max()
    last = df[df["bins"] == last_bin]
    last_sorted = last.sort_values("stats", ascending=False)
    return last_sorted.head(5)[[cluster_key, "stats"]].values.tolist()


# train: slideseqv2
adata_train = sq.datasets.slideseqv2()
print("train (slideseqv2):", top_clustered_cluster(adata_train, "cluster"))

# test: seqfish
adata_test = sq.datasets.seqfish()
print("test (seqfish):", top_clustered_cluster(adata_test, "celltype_mapped_refined"))
