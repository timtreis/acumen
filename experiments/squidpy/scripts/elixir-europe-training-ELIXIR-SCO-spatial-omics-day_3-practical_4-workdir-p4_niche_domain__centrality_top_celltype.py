import squidpy as sq


def top_degree_centrality(adata, cluster_key):
    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="generic")
    sq.gr.centrality_scores(adata, cluster_key=cluster_key)
    df = adata.uns[f"{cluster_key}_centrality_scores"]
    return df["degree_centrality"].idxmax()


# train: mibitof data, sample point16
adata_train = sq.datasets.mibitof()
sub_train = adata_train[adata_train.obs["library_id"] == "point16"].copy()
print("TRAIN answer:", top_degree_centrality(sub_train, "Cluster"))

# test: imc data
adata_test = sq.datasets.imc()
print("TEST answer:", top_degree_centrality(adata_test, "cell type"))
