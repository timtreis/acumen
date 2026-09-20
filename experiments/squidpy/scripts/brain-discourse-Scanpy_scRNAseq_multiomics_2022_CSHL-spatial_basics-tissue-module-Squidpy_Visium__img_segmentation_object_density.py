"""
Reproduces the answers for task `img_segmentation_object_density`.

Analysis: smooth and segment the tissue image into individual objects with a
watershed algorithm, count how many segmented objects fall within each
Visium spot, and determine which annotated tissue cluster has the highest
average object count per spot.

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

    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed")

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
    ranking = (
        adata.obs.assign(v=df["segmentation_label"].values)
        .groupby("cluster", observed=True)["v"]
        .mean()
        .sort_values(ascending=False)
    )

    print(split, "top cluster:", ranking.index[0])
    print(ranking.head(5))
    print()
