import os

os.environ["TMPDIR"] = "/tmp"

import squidpy as sq

# train: imc dataset, cluster key "cell type"
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
sq.gr.centrality_scores(adata, cluster_key="cell type")
df = adata.uns["cell type_centrality_scores"]
print("train answer:", df.sort_values("degree_centrality", ascending=False).index[0])

# test: mibitof dataset, cluster key "Cluster"
adata2 = sq.datasets.mibitof()
sq.gr.spatial_neighbors(adata2, library_key="library_id", coord_type="generic", delaunay=True)
sq.gr.centrality_scores(adata2, cluster_key="Cluster")
df2 = adata2.uns["Cluster_centrality_scores"]
print("test answer:", df2.sort_values("degree_centrality", ascending=False).index[0])
