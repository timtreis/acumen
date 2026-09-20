import numpy as np
import squidpy as sq

# --- train variant: imc dataset, radius = 20 ---
adata_train = sq.datasets.imc()
sq.gr.spatial_neighbors(adata_train, coord_type="generic", radius=20)
conn_train = adata_train.obsp["spatial_connectivities"]
deg_train = np.asarray((conn_train > 0).sum(1)).flatten()
print("train (imc, radius=20) avg degree:", round(float(deg_train.mean()), 2))

# --- test variant: seqfish dataset, radius = 0.05 ---
adata_test = sq.datasets.seqfish()
sq.gr.spatial_neighbors(adata_test, coord_type="generic", radius=0.05)
conn_test = adata_test.obsp["spatial_connectivities"]
deg_test = np.asarray((conn_test > 0).sum(1)).flatten()
print("test (seqfish, radius=0.05) avg degree:", round(float(deg_test.mean()), 2))
