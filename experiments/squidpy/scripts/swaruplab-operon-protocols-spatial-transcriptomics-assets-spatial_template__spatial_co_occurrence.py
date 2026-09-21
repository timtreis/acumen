import numpy as np
import squidpy as sq
import warnings
warnings.filterwarnings("ignore")

adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type")
res = adata.uns["cell type_co_occurrence"]
occ = res["occ"]
cats = adata.obs["cell type"].cat.categories.tolist()

for ref in ["T cells", "p53+ EGFR+ tumor cell"]:
    ridx = cats.index(ref)
    vals = occ[ridx, :, 0].copy()
    vals[ridx] = -1
    k = int(np.argmax(vals))
    print(f"reference={ref!r} -> top co-occurring partner at short range: {cats[k]!r} (ratio={vals[k]:.3f})")
