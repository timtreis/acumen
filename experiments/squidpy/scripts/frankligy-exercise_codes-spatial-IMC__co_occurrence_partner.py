import squidpy as sq
import pandas as pd

adata = sq.datasets.imc()
out, interval = sq.gr.co_occurrence(adata, cluster_key="cell type", copy=True)
cats = list(adata.obs["cell type"].cat.categories)

train_target = "basal CK tumor cell"
test_target = "p53+ EGFR+ tumor cell"

for target in (train_target, test_target):
    idx = cats.index(target)
    maxvals = out[idx, :, :].max(axis=1)
    s = pd.Series(maxvals, index=cats).drop(target).sort_values(ascending=False)
    print(target, "->", s.index[0])
