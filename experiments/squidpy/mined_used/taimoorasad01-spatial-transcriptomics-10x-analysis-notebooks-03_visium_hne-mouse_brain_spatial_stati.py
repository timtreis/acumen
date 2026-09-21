# mined from: https://github.com/taimoorasad01/spatial-transcriptomics-10x-analysis/blob/6584a774671de3bf2977164d4434c8b613b4616d/notebooks/03_visium_hne/mouse_brain_spatial_statistics.ipynb
# symbols: squidpy.datasets.visium_hne_adata, squidpy.datasets.visium_hne_image, squidpy.gr.co_occurrence, squidpy.gr.ligrec, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.im.calculate_image_features, squidpy.pl.co_occurrence, squidpy.pl.ligrec, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
# %matplotlib inline

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# %%
img = sq.datasets.visium_hne_image()
adata = sq.datasets.visium_hne_adata()

# %%
sq.pl.spatial_scatter(adata, color="cluster")
plt.savefig("../results/03_visium_hne/01_spatial_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
import matplotlib.pyplot as plt

for scale in [1.0, 2.0]:
    feature_name = f"features_summary_scale{scale}"
    sq.im.calculate_image_features(
        adata,
        img.compute(),
        features="summary",
        key_added=feature_name,
        n_jobs=4,
        scale=scale,
    )

adata.obsm["features"] = pd.concat(
    [adata.obsm[f] for f in adata.obsm.keys() if "features_summary" in f],
    axis="columns",
)
adata.obsm["features"].columns = ad.utils.make_index_unique(
    adata.obsm["features"].columns
)

# %%
def cluster_features(features: pd.DataFrame, like=None) -> pd.Series:
    """Calculate leiden clustering of features."""
    if like is not None:
        features = features.filter(like=like)
    adata = ad.AnnData(features)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=min(10, features.shape[1] - 1))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    return adata.obs["leiden"]

adata.obs["features_cluster"] = cluster_features(adata.obsm["features"], like="summary")
sq.pl.spatial_scatter(adata, color=["features_cluster", "cluster"])
plt.savefig("../results/03_visium_hne/01_spatial_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cluster")
sq.pl.nhood_enrichment(adata, cluster_key="cluster")
plt.savefig("../results/03_visium_hne/02_nhood_enrichment.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
sq.gr.co_occurrence(adata, cluster_key="cluster")
sq.pl.co_occurrence(
    adata,
    cluster_key="cluster",
    clusters="Hippocampus",
    figsize=(8, 4),
)
plt.savefig("../results/03_visium_hne/03_co_occurrence.png", dpi=150, bbox_inches="tight")
plt.show()

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
plt.savefig("../results/03_visium_hne/04_ligrec.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
genes = adata[:, adata.var.highly_variable].var_names.values[:1000]
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    genes=genes,
    n_perms=100,
    n_jobs=1,
)
adata.uns["moranI"].head(10)

# %%
sq.pl.spatial_scatter(adata, color=["Olfm1", "Plp1", "Itpka", "cluster"])
plt.savefig("../results/03_visium_hne/05_moranI_genes.png", dpi=150, bbox_inches="tight")
plt.show()
