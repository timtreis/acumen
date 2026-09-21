import squidpy as sq

# --- train: imc dataset ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.centrality_scores(adata, "cell type")
df = adata.uns["cell type_centrality_scores"]
print("TRAIN (imc) max average_clustering:", df["average_clustering"].idxmax(), df["average_clustering"].max())

# --- test: mibitof dataset (multiple images -> library_key) ---
adata2 = sq.datasets.mibitof()
sq.gr.spatial_neighbors(adata2, library_key="library_id")
sq.gr.centrality_scores(adata2, "Cluster")
df2 = adata2.uns["Cluster_centrality_scores"]
print("TEST (mibitof) max average_clustering:", df2["average_clustering"].idxmax(), df2["average_clustering"].max())
