import squidpy as sq

adata = sq.datasets.slideseqv2()

# train: named cell-type annotation
res_cluster = sq.gr.ripley(adata, cluster_key="cluster", mode="L", copy=True)
top_cluster = res_cluster["L_stat"].groupby("cluster")["stats"].max().sort_values(ascending=False)
print("train (cell-type annotation) top:", top_cluster.index[0], round(top_cluster.iloc[0], 2))

# test: numeric unsupervised (leiden) clusters
res_leiden = sq.gr.ripley(adata, cluster_key="leiden", mode="L", copy=True)
top_leiden = res_leiden["L_stat"].groupby("leiden")["stats"].max().sort_values(ascending=False)
print("test (numeric unsupervised clusters) top:", top_leiden.index[0], round(top_leiden.iloc[0], 2))

# train answer: "CA1_CA2_CA3_Subiculum"
# test answer:  "0"
