"""
Ground truth for task 'pc3_top_cluster'.

The napari tutorial demonstrates overlaying a searchable feature from
adata.obsm (the third column of X_pca, i.e. the third principal component
of the gene-expression PCA) on top of the tissue image, alongside the
precomputed tissue-region annotation stored in adata.obs['cluster']. The
reproducible fact behind that overlay is: which annotated region has the
highest average value along that third principal component.
"""
import pandas as pd
import squidpy as sq


def top_region_by_pc3(adata):
    pc3 = adata.obsm["X_pca"][:, 2]
    df = pd.DataFrame({"cluster": adata.obs["cluster"].values, "pc3": pc3})
    means = df.groupby("cluster")["pc3"].mean().sort_values(ascending=False)
    return means


# train variant
adata_hne = sq.datasets.visium_hne_adata()
means_hne = top_region_by_pc3(adata_hne)
print("hne (train) mean PC3 by region:\n", means_hne)
print("hne (train) top region:", means_hne.index[0])

# test variant
adata_fluo = sq.datasets.visium_fluo_adata()
means_fluo = top_region_by_pc3(adata_fluo)
print("fluo (test) mean PC3 by region:\n", means_fluo)
print("fluo (test) top region:", means_fluo.index[0])
