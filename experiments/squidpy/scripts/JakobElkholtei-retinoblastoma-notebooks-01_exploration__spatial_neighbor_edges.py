import squidpy as sq

# train: visium_hne_adata
adata_train = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata_train)
train_edges = adata_train.obsp["spatial_connectivities"].nnz
print("train (visium_hne_adata) edges:", train_edges)

# test: visium_fluo_adata
adata_test = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata_test)
test_edges = adata_test.obsp["spatial_connectivities"].nnz
print("test (visium_fluo_adata) edges:", test_edges)
