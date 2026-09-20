import squidpy as sq


def most_clustered(adata, cluster_key):
    res = sq.gr.ripley(adata, cluster_key=cluster_key, mode="L", seed=0, copy=True)
    df = res["L_stat"]
    max_bin = df["bins"].max()
    last = df[df["bins"] == max_bin].sort_values("stats", ascending=False)
    return last.iloc[0][cluster_key], last.iloc[0]["stats"]


if __name__ == "__main__":
    # train: imc data
    adata_imc = sq.datasets.imc()
    print("train (imc):", most_clustered(adata_imc, "cell type"))

    # test: mibitof, sample point8
    adata_full = sq.datasets.mibitof()
    adata8 = adata_full[adata_full.obs["library_id"] == "point8"].copy()
    adata8.obs["Cluster"] = adata8.obs["Cluster"].cat.remove_unused_categories()
    print("test (mibitof point8):", most_clustered(adata8, "Cluster"))
