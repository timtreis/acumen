import squidpy as sq


def top_degree_centrality_celltype(adata, cluster_key, library_key=None):
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, library_key=library_key)
    sq.gr.centrality_scores(adata, cluster_key=cluster_key)
    df = adata.uns[f"{cluster_key}_centrality_scores"]
    return df["degree_centrality"].idxmax()


# TRAIN: imc dataset
adata_train = sq.datasets.imc()
print("TRAIN answer:", top_degree_centrality_celltype(adata_train, cluster_key="cell type"))

# TEST: mibitof dataset (multiple imaging points -> use library_key)
adata_test = sq.datasets.mibitof()
print(
    "TEST answer:",
    top_degree_centrality_celltype(adata_test, cluster_key="Cluster", library_key="library_id"),
)
