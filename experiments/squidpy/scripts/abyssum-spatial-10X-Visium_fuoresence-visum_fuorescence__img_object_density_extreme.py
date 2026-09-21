"""
Task: img_object_density_extreme
Segment the tissue image into discrete objects (nuclei-like blobs) and, for each spot,
count how many objects fall inside it. Group by the dataset's existing transcriptomic
cluster label and report the cluster with the HIGHEST (train) / LOWEST (test) average
object count per spot. Dataset: visium_fluo_adata_crop / visium_fluo_image_crop.
"""

import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

sq.im.segment(img=img, layer="image", chunks=1000)  # default method="watershed", channel=0

features_kwargs = {"segmentation": {"label_layer": "segmented_watershed"}}
sq.im.calculate_image_features(
    adata,
    img,
    features="segmentation",
    layer="image",
    key_added="features_segmentation",
    n_jobs=1,
    features_kwargs=features_kwargs,
    show_progress_bar=False,
)
df = adata.obsm["features_segmentation"]
adata.obs["seg_label"] = df["segmentation_label"].values
means = adata.obs.groupby("cluster")["seg_label"].mean().sort_values()
print(means)
print("train answer (highest):", means.index[-1])
print("test answer (lowest):", means.index[0])
