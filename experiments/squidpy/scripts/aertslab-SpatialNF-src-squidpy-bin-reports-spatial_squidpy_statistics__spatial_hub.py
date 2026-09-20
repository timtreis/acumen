"""Task: spatial_hub
Find the cell type that is the most spatially connected 'hub' (highest degree
centrality in the spatial neighbor graph) in a bundled dataset.
"""
import squidpy as sq


def top_degree_centrality(adata, key, library_key=None):
    if library_key is not None:
        sq.gr.spatial_neighbors(adata, library_key=library_key)
    else:
        sq.gr.spatial_neighbors(adata)
    sq.gr.centrality_scores(adata, cluster_key=key)
    cs = adata.uns[f"{key}_centrality_scores"]
    return cs["degree_centrality"].idxmax(), cs["degree_centrality"].max()


# --- train: imc dataset ---
adata_imc = sq.datasets.imc()
top_imc = top_degree_centrality(adata_imc, "cell type")
print("TRAIN (imc):", top_imc)
# -> ('apoptotic tumor cell', ~0.83)

# --- test: mibitof dataset ---
adata_mibi = sq.datasets.mibitof()
top_mibi = top_degree_centrality(adata_mibi, "Cluster", library_key="library_id")
print("TEST (mibitof):", top_mibi)
# -> ('Tcell_CD4', ~0.44)
