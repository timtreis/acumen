"""Ground-truth script for task: spatial_graph_edges (train + test)."""
import squidpy as sq


def total_edges(adata):
    sq.gr.spatial_neighbors(adata)
    conn = adata.obsp["spatial_connectivities"]
    return conn.nnz // 2  # symmetric matrix -> undirected edge count


# --- train variant ---
adata_hne = sq.datasets.visium_hne_adata()
print("train: dataset=visium_hne, edges =", total_edges(adata_hne))

# --- test variant ---
adata_fluo = sq.datasets.visium_fluo_adata()
print("test: dataset=visium_fluo, edges =", total_edges(adata_fluo))
