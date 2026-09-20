import squidpy as sq

# TRAIN: visium_hne_adata, most connected spatial domain
adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)
sq.gr.centrality_scores(adata, cluster_key="cluster")
df = adata.uns["cluster_centrality_scores"]
top_train = df.sort_values("degree_centrality", ascending=False).index[0]
print("TRAIN most central domain:", top_train)
print(df.sort_values("degree_centrality", ascending=False).head())

# TEST: imc dataset, most connected cell type
adata2 = sq.datasets.imc()
sq.gr.spatial_neighbors(adata2)
sq.gr.centrality_scores(adata2, cluster_key="cell type")
df2 = adata2.uns["cell type_centrality_scores"]
top_test = df2.sort_values("degree_centrality", ascending=False).index[0]
print("TEST most central cell type:", top_test)
print(df2.sort_values("degree_centrality", ascending=False).head())
