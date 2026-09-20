"""Ground truth for the 'fixed_radius_neighbors' task (train + test variants).

Goal: build a spatial neighbor graph that connects each cell to every other
cell within a fixed physical distance (radius) of the given cells, rather
than a fixed number of nearest neighbors, and report the average number of
neighbors per cell (mean graph degree), rounded to two decimals.

Train variant: imc dataset, radius = 20.0 (dataset coordinate units).
Test variant:  seqfish dataset, radius = 0.05 (dataset coordinate units).
"""
import numpy as np
import squidpy as sq


def mean_degree(adata, radius):
    sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius)
    conn = adata.obsp["spatial_connectivities"]
    degree = np.asarray(conn.sum(axis=1)).flatten()
    return degree.mean()


# --- train: imc, radius 20.0 ---
adata_train = sq.datasets.imc()
train_mean_degree = mean_degree(adata_train, 20.0)
print("train mean degree:", train_mean_degree)
print("train answer:", round(train_mean_degree, 2))

# --- test: seqfish, radius 0.05 ---
adata_test = sq.datasets.seqfish()
test_mean_degree = mean_degree(adata_test, 0.05)
print("test mean degree:", test_mean_degree)
print("test answer:", round(test_mean_degree, 2))
