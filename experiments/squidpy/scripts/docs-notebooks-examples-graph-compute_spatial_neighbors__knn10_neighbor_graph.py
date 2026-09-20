import squidpy as sq

# train: imc, k=10 nearest neighbors
a_train = sq.datasets.imc()
sq.gr.spatial_neighbors(a_train, coord_type="generic", n_neighs=10)
print("train (imc) nnz:", a_train.obsp["spatial_connectivities"].nnz)

# test: seqfish, k=10 nearest neighbors
a_test = sq.datasets.seqfish()
sq.gr.spatial_neighbors(a_test, coord_type="generic", n_neighs=10)
print("test (seqfish) nnz:", a_test.obsp["spatial_connectivities"].nnz)
