import squidpy as sq

# train: seqfish, delaunay triangulation
a_train = sq.datasets.seqfish()
sq.gr.spatial_neighbors(a_train, coord_type="generic", delaunay=True)
print("train (seqfish) nnz:", a_train.obsp["spatial_connectivities"].nnz)

# test: imc, delaunay triangulation
a_test = sq.datasets.imc()
sq.gr.spatial_neighbors(a_test, coord_type="generic", delaunay=True)
print("test (imc) nnz:", a_test.obsp["spatial_connectivities"].nnz)
