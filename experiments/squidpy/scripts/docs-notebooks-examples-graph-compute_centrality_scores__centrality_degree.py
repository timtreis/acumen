import squidpy as sq

# --- train: imc dataset ---
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.centrality_scores(adata, "cell type")
df = adata.uns["cell type_centrality_scores"]
print("TRAIN (imc) min degree_centrality:", df["degree_centrality"].idxmin(), df["degree_centrality"].min())

# --- test: mibitof dataset (multiple images -> library_key) ---
adata2 = sq.datasets.mibitof()
sq.gr.spatial_neighbors(adata2, library_key="library_id")
sq.gr.centrality_scores(adata2, "Cluster")
df2 = adata2.uns["Cluster_centrality_scores"]
print("TEST (mibitof) min degree_centrality:", df2["degree_centrality"].idxmin(), df2["degree_centrality"].min())
