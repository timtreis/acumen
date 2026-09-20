import os

os.environ.setdefault("TMPDIR", "/tmp/mf_work")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import squidpy as sq

adata = sq.datasets.merfish()


def top_gene(bregma):
    a = adata[adata.obs.Bregma == bregma].copy()
    sq.gr.spatial_neighbors(a, coord_type="generic")
    sq.gr.spatial_autocorr(a, mode="moran", n_jobs=1)
    return a.uns["moranI"].index[0], a.uns["moranI"].iloc[0]["I"]

# train variant: the Bregma = -9 slice used in the tutorial
print("train (Bregma=-9):", top_gene(-9.0))

# test variant: the Bregma = 16 slice
print("test (Bregma=16):", top_gene(16.0))
