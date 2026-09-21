"""
Task: img_intensity_cluster
Compute per-spot summary image features (mean pixel intensity of the tissue image
underneath each spot, averaged across all image channels) on the visium_fluo_adata_crop /
visium_fluo_image_crop dataset. Report which transcriptomic cluster has the HIGHEST
(train) / LOWEST (test) average value.
"""

import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

sq.im.calculate_image_features(
    adata,
    img,
    layer="image",
    key_added="summary_feats",
    n_jobs=1,
    features="summary",
    show_progress_bar=False,
)
df = adata.obsm["summary_feats"]
mean_cols = [c for c in df.columns if c.endswith("_mean") and "ch-" in c]
adata.obs["overall_mean"] = df[mean_cols].mean(axis=1).values
means = adata.obs.groupby("cluster")["overall_mean"].mean().sort_values()
print(means)
print("train answer (highest):", means.index[-1])
print("test answer (lowest):", means.index[0])
