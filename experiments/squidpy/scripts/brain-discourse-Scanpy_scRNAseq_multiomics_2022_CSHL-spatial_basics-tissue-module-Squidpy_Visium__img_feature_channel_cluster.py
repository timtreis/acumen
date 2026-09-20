"""
Reproduces the answers for task `img_feature_channel_cluster`.

Analysis: compute per-spot summary image-intensity features from the tissue
image paired with a Visium dataset, then determine which annotated tissue
cluster has the highest average intensity in the second image channel
(channel index 1).

train -> visium_fluo_adata_crop / visium_fluo_image_crop
test  -> visium_hne_adata_crop  / visium_hne_image_crop
"""

import os

os.environ.setdefault("TMPDIR", "/tmp")

import squidpy as sq

DATASETS = {
    "train": (sq.datasets.visium_fluo_adata_crop, sq.datasets.visium_fluo_image_crop),
    "test": (sq.datasets.visium_hne_adata_crop, sq.datasets.visium_hne_image_crop),
}

for split, (adata_fn, img_fn) in DATASETS.items():
    adata = adata_fn()
    img = img_fn()

    sq.im.calculate_image_features(
        adata,
        img,
        features="summary",
        layer="image",
        key_added="summary",
        n_jobs=1,
        show_progress_bar=False,
    )

    df = adata.obsm["summary"]
    col = "summary_ch-1_mean"
    ranking = (
        adata.obs.assign(v=df[col].values)
        .groupby("cluster", observed=True)["v"]
        .mean()
        .sort_values(ascending=False)
    )

    print(split, "top cluster:", ranking.index[0])
    print(ranking.head(5))
    print()
