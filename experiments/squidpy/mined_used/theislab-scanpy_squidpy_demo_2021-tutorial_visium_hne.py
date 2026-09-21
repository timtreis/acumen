# mined from: https://github.com/theislab/scanpy_squidpy_demo_2021/blob/7dc7b1a9b341d91bca2f31cfc7e9d529a5b04a52/tutorial_visium_hne.ipynb
# symbols: squidpy.datasets.visium_hne_adata, squidpy.datasets.visium_hne_image, squidpy.gr.co_occurrence, squidpy.gr.ligrec, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.im.calculate_image_features, squidpy.pl.co_occurrence, squidpy.pl.ligrec, squidpy.pl.nhood_enrichment

# %%
# %matplotlib inline

# %%
# !pip install squidpy
# !pip install numba==0.52

# %%
import scanpy as sc
import anndata as ad
import squidpy as sq

import numpy as np
import pandas as pd

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
img = sq.datasets.visium_hne_image()
adata = sq.datasets.visium_hne_adata()

# %%
sc.pl.spatial(adata, color="cluster")

# %%
# calculate features for different scales (higher value means more context)
for scale in [1.0, 2.0]:
    feature_name = f"features_summary_scale{scale}"
    sq.im.calculate_image_features(
        adata,
        img,
        features="summary",
        key_added=feature_name,
        n_jobs=1,
        scale=scale,
    )


# combine features in one dataframe
adata.obsm["features"] = pd.concat(
    [adata.obsm[f] for f in adata.obsm.keys() if "features_summary" in f], axis="columns"
)
# make sure that we have no duplicated feature names in the combined table
adata.obsm["features"].columns = ad.utils.make_index_unique(adata.obsm["features"].columns)

# %%
# helper function returning a clustering
def cluster_features(features: pd.DataFrame, like=None):
    """Calculate leiden clustering of features.

    Specify filter of features using `like`.
    """
    # filter features
    if like is not None:
        features = features.filter(like=like)
    # create temporary adata to calculate the clustering
    adata = ad.AnnData(features)
    # important - feature values are not scaled, so need to scale them before PCA
    sc.pp.scale(adata)
    # calculate leiden clustering
    sc.pp.pca(adata, n_comps=min(10, features.shape[1] - 1))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)

    return adata.obs["leiden"]


# calculate feature clusters
adata.obs["features_cluster"] = cluster_features(adata.obsm["features"], like="summary")

# compare feature and gene clusters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.pl.spatial(adata, color=["features_cluster", "cluster"])

# %%
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cluster")
sq.pl.nhood_enrichment(adata, cluster_key="cluster")

# %%
sq.gr.co_occurrence(adata, cluster_key="cluster")
sq.pl.co_occurrence(
    adata,
    cluster_key="cluster",
    clusters="Hippocampus",
    figsize=(8, 4),
)

# %%
sq.gr.ligrec(
    adata,
    n_perms=100,
    cluster_key="cluster",
)
sq.pl.ligrec(
    adata,
    cluster_key="cluster",
    source_groups="Hippocampus",
    target_groups=["Pyramidal_layer", "Pyramidal_layer_dentate_gyrus"],
    means_range=(3, np.inf),
    alpha=1e-4,
    swap_axes=True,
)

# %%
genes = adata[:, adata.var.highly_variable].var_names.values
sq.gr.moran(
    adata,
    genes=genes,
)

# %%
adata.uns["moranI"].head(10)

# %%
sc.pl.spatial(adata, color=["Olfm1", "Plp1", "Itpka", "cluster"])
