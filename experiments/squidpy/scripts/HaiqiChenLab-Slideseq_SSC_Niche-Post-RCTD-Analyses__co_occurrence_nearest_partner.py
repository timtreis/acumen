import squidpy as sq
import numpy as np
import pandas as pd


def nearest_bin_top_partner(types, ref):
    adata = sq.datasets.slideseqv2()
    a = adata[adata.obs["cluster"].isin(types)].copy()
    a.obs["cluster"] = a.obs["cluster"].cat.remove_unused_categories()
    sq.gr.co_occurrence(a, "cluster")
    res = a.uns["cluster_co_occurrence"]
    occ = res["occ"]
    cats = a.obs["cluster"].cat.categories.tolist()
    ri = cats.index(ref)
    probs = pd.Series(occ[ri, :, 0], index=cats)
    other = probs.drop(ref)
    return other.idxmax(), round(float(other.max()), 4)


# train variant
train_types = [
    "CA1_CA2_CA3_Subiculum",
    "DentatePyramids",
    "Subiculum_Entorhinal_cl2",
    "Subiculum_Entorhinal_cl3",
]
print("train:", nearest_bin_top_partner(train_types, "DentatePyramids"))

# test variant
test_types = [
    "Astrocytes",
    "Oligodendrocytes",
    "Polydendrocytes",
    "Microglia",
    "Endothelial_Stalk",
    "Endothelial_Tip",
    "Mural",
]
print("test:", nearest_bin_top_partner(test_types, "Microglia"))
