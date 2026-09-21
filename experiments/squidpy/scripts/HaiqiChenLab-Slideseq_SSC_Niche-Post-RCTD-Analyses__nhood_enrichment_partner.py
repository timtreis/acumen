import squidpy as sq
import numpy as np
import pandas as pd


def top_partner(types, ref, seed=0):
    adata = sq.datasets.slideseqv2()
    a = adata[adata.obs["cluster"].isin(types)].copy()
    a.obs["cluster"] = a.obs["cluster"].cat.remove_unused_categories()
    sq.gr.spatial_neighbors(a)
    sq.gr.nhood_enrichment(a, cluster_key="cluster", seed=seed, show_progress_bar=False)
    cats = a.obs["cluster"].cat.categories.tolist()
    z = np.array(a.uns["cluster_nhood_enrichment"]["zscore"])
    df = pd.DataFrame(z, index=cats, columns=cats)
    row = df.loc[ref].drop(ref)
    return row.idxmax(), round(float(row.max()), 2)


# train variant
train_types = [
    "CA1_CA2_CA3_Subiculum",
    "DentatePyramids",
    "Subiculum_Entorhinal_cl2",
    "Subiculum_Entorhinal_cl3",
]
print("train:", top_partner(train_types, "CA1_CA2_CA3_Subiculum"))

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
print("test:", top_partner(test_types, "Endothelial_Tip"))
