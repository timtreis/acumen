import squidpy as sq

# train: imc dataset
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.centrality_scores(adata, cluster_key="cell type")
df = adata.uns["cell type_centrality_scores"]
print("train answer:", df["average_clustering"].idxmax())

# test: seqfish dataset
adata2 = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata2)
sq.gr.centrality_scores(adata2, cluster_key="celltype_mapped_refined")
df2 = adata2.uns["celltype_mapped_refined_centrality_scores"]
print("test answer:", df2["average_clustering"].idxmax())
