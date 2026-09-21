import squidpy as sq
import numpy as np

a = sq.datasets.mibitof()
target = "Tcell_CD8"

for lib in ["point16", "point23"]:
    sub = a[a.obs.library_id == lib].copy()
    occ, interval = sq.gr.co_occurrence(sub, cluster_key="Cluster", copy=True)
    cats = sub.obs["Cluster"].cat.categories.tolist()
    ti = cats.index(target)
    vals = occ[ti, :, 0].copy()
    vals[ti] = -np.inf
    j = int(np.argmax(vals))
    print(lib, "-> most co-occurring with", target, ":", cats[j], round(float(vals[j]), 3))

# train (point16) answer: Tcell_CD4
# test (point23) answer: Imm_other
