"""
Reproduces the answer for the `visium_fluo_segmentation_cell_count` task.

Segments nuclei in the DAPI channel of the visium_fluo fluorescence image (smoothing
+ watershed), then uses squidpy's segmentation image features to estimate the number
of cells per Visium spot. Averages this estimate within each annotated tissue-region
('cluster' obs column) and reports the region with the highest / lowest average.
"""

import squidpy as sq


def main():
    img = sq.datasets.visium_fluo_image_crop()
    adata = sq.datasets.visium_fluo_adata_crop()

    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0, chunks=1000)

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

    feat = adata.obsm["features_segmentation"].copy()
    feat["cluster"] = adata.obs["cluster"].values

    # segmentation_label counts the number of segmented nuclei per spot
    per_cluster = feat.groupby("cluster")["segmentation_label"].mean().sort_values()
    print(per_cluster)
    print("train answer (highest avg cell count):", per_cluster.idxmax())
    print("test answer (lowest avg cell count):", per_cluster.idxmin())


if __name__ == "__main__":
    main()
