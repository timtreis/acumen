# mined from: https://github.com/taimoorasad01/spatial-transcriptomics-10x-analysis/blob/6584a774671de3bf2977164d4434c8b613b4616d/notebooks/02_visium_fluorescence/mouse_brain_fluorescence_imaging.ipynb
# symbols: squidpy.datasets.visium_fluo_adata_crop, squidpy.datasets.visium_fluo_image_crop, squidpy.im.calculate_image_features, squidpy.im.process, squidpy.im.segment, squidpy.pl.extract, squidpy.pl.spatial_scatter

# %%
# %matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import squidpy as sq

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# %%
# Load pre-processed dataset and tissue image
img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

# %%
sq.pl.spatial_scatter(adata, color="cluster")
plt.savefig("../results/02_visium_fluo/01_spatial_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
img.show(channelwise=True)
plt.savefig("../results/02_visium_fluo/02_fluorescence_channels.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
sq.im.process(
    img=img,
    layer="image",
    method="smooth",
)

sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0, chunks=1000)

# %%
# Visualize segmentation result
fig, ax = plt.subplots(1, 2)
img_crop = img.crop_corner(2000, 2000, size=500)
img_crop.show(layer="image", channel=0, ax=ax[0])
img_crop.show(
    layer="segmented_watershed",
    channel=0,
    ax=ax[1],
)
plt.savefig("../results/02_visium_fluo/03_segmentation.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
features_kwargs = {"segmentation": {"label_layer": "segmented_watershed"}}

sq.im.calculate_image_features(
    adata,
    img,
    features="segmentation",
    layer="image",
    key_added="features_segmentation",
    n_jobs=1,
    features_kwargs=features_kwargs,
)

# %%
sq.pl.spatial_scatter(
    sq.pl.extract(adata, "features_segmentation"),
    color=[
        "segmentation_label",
        "cluster",
        "segmentation_ch-0_mean_intensity_mean",
        "segmentation_ch-1_mean_intensity_mean",
    ],
    frameon=False,
    ncols=2,
)
plt.savefig("../results/02_visium_fluo/04_segmentation_features.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
params = {
    "features_orig": {
        "features": ["summary", "texture", "histogram"],
        "scale": 1.0,
        "mask_circle": True,
    },
    "features_context": {"features": ["summary", "histogram"], "scale": 1.0},
    "features_lowres": {"features": ["summary", "histogram"], "scale": 0.25},
}

for feature_name, cur_params in params.items():
    sq.im.calculate_image_features(
        adata, img, layer="image", key_added=feature_name, n_jobs=1, **cur_params
    )

adata.obsm["features"] = pd.concat(
    [adata.obsm[f] for f in params.keys()], axis="columns"
)
adata.obsm["features"].columns = ad.utils.make_index_unique(
    adata.obsm["features"].columns
)

# %%
def cluster_features(features: pd.DataFrame, like=None):
    """Calculate leiden clustering of features."""
    if like is not None:
        features = features.filter(like=like)
    adata = ad.AnnData(features)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=min(10, features.shape[1] - 1))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    return adata.obs["leiden"]

# %%
adata.obs["features_summary_cluster"] = cluster_features(
    adata.obsm["features"], like="summary"
)
adata.obs["features_histogram_cluster"] = cluster_features(
    adata.obsm["features"], like="histogram"
)
adata.obs["features_texture_cluster"] = cluster_features(
    adata.obsm["features"], like="texture"
)

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sq.pl.spatial_scatter(
    adata,
    color=[
        "features_summary_cluster",
        "features_histogram_cluster",
        "features_texture_cluster",
        "cluster",
    ],
    ncols=3,
)
plt.savefig("../results/02_visium_fluo/05_feature_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
