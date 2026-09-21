import os
os.environ.setdefault("TMPDIR", "/tmp")

import squidpy as sq

adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
df = sq.gr.centrality_scores(adata, cluster_key="cell type", copy=True)

print("highest average_clustering:", df["average_clustering"].idxmax(), df["average_clustering"].max())
print("lowest average_clustering:", df["average_clustering"].idxmin(), df["average_clustering"].min())
