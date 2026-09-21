import squidpy as sq
import pandas as pd

adata = sq.datasets.slideseqv2()
cats = adata.obs["cluster"].cat.categories.tolist()
idx = {c: i for i, c in enumerate(cats)}

sq.gr.co_occurrence(adata, cluster_key="cluster", spatial_key="spatial")
occ = adata.uns["cluster_co_occurrence"]["occ"]

for target in ["Neurogenesis", "Polydendrocytes"]:
    i = idx[target]
    probs_closest_bin = occ[i, :, 0]
    s = pd.Series(probs_closest_bin, index=cats).drop(target).sort_values(ascending=False)
    print(target, "-> highest co-occurrence probability at shortest range:", s.index[0], round(s.iloc[0], 3))

# train answer: target "Neurogenesis" -> "Ependymal"
# test answer:  target "Polydendrocytes" -> "Oligodendrocytes"
