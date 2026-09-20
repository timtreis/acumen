"""Ground-truth script for task mibitof_centrality.

Goal: which annotated cell type is the best-connected hub (highest degree
centrality) of the spatial neighbor graph, per imaging region (point) of the
mibitof dataset.

train -> point8
test  -> point16
"""

import squidpy as sq

adata = sq.datasets.mibitof()

for point_id in ["point8", "point16"]:
    sub = adata[adata.obs["library_id"] == point_id].copy()
    sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
    sq.gr.centrality_scores(sub, cluster_key="Cluster")
    df = sub.uns["Cluster_centrality_scores"].sort_values("degree_centrality", ascending=False)
    print(point_id)
    print(df)
    print("TOP:", df.index[0])
    print()
