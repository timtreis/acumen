import os

os.environ["TMPDIR"] = "/tmp"

import squidpy as sq

if __name__ == "__main__":
    # train: visium_hne_adata (mouse brain Visium); spatial_autocorr defaults to
    # highly-variable genes when adata.var["highly_variable"] is present
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=1, seed=0)
    print("train answer:", adata.uns["moranI"].index[0])

    # test: imc dataset
    adata2 = sq.datasets.imc()
    sq.gr.spatial_neighbors(adata2, coord_type="generic", delaunay=True)
    sq.gr.spatial_autocorr(adata2, mode="moran", n_perms=100, n_jobs=1, seed=0)
    print("test answer:", adata2.uns["moranI"].index[0])
