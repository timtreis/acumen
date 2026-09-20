import os
import sys

# multiprocessing's Manager() binds a unix socket under TMPDIR; re-exec with a
# short TMPDIR so its path doesn't exceed the AF_UNIX length limit.
_short_tmp = "/tmp/mf_work"
if os.environ.get("TMPDIR") != _short_tmp:
    os.makedirs(_short_tmp, exist_ok=True)
    os.environ["TMPDIR"] = _short_tmp
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import squidpy as sq


def strongest_partner(z, cats, query):
    i = cats.index(query)
    row = z[i].copy()
    row[i] = -np.inf
    j = int(np.argmax(row))
    return cats[j], row[j]


if __name__ == "__main__":
    adata = sq.datasets.merfish()

    # Build a single neighbor graph over the full 3D stack (all Bregma slices)
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial3d")
    sq.gr.nhood_enrichment(adata, cluster_key="Cell_class", n_jobs=1)

    z = adata.uns["Cell_class_nhood_enrichment"]["zscore"]
    cats = list(adata.obs["Cell_class"].cat.categories)

    # train variant: microglia
    print("train (Microglia):", strongest_partner(z, cats, "Microglia"))

    # test variant: astrocyte
    print("test (Astrocyte):", strongest_partner(z, cats, "Astrocyte"))
