import squidpy as sq

# train: visium_fluo
a_train = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(a_train)
print("train (visium_fluo) nnz:", a_train.obsp["spatial_connectivities"].nnz)

# test: visium_hne
a_test = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(a_test)
print("test (visium_hne) nnz:", a_test.obsp["spatial_connectivities"].nnz)
