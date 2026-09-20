"""Ground-truth script for task mibitof_ripley_clustering.

Goal: which annotated cell type shows the strongest spatial clustering
(highest Ripley's L statistic at the largest distance scale) in a given
imaging region (point) of the mibitof dataset.

train -> point8
test  -> point23
"""

import squidpy as sq

adata = sq.datasets.mibitof()

for point_id in ["point8", "point23"]:
    sub = adata[adata.obs["library_id"] == point_id].copy()
    sq.gr.ripley(sub, cluster_key="Cluster", mode="L", seed=0)
    df = sub.uns["Cluster_ripley_L"]["L_stat"]
    max_bin = df["bins"].max()
    at_max = df[df["bins"] == max_bin].sort_values("stats", ascending=False)

    print(point_id)
    print(at_max)
    print("TOP:", at_max.iloc[0]["Cluster"])
    print()
