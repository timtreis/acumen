import squidpy as sq

# train
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
print("train (imc) edges:", adata.obsp["spatial_connectivities"].nnz)

# test
adata2 = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata2, coord_type="generic", delaunay=True)
print("test (seqfish) edges:", adata2.obsp["spatial_connectivities"].nnz)
