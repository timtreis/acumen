"""
Reproduces the answer for the `visium_fluo_marker_channel_intensity` task.

Segments nuclei in the DAPI channel of the visium_fluo fluorescence image (smoothing
+ watershed), then uses squidpy's segmentation image features to get the mean
per-nucleus fluorescence intensity for the anti-NEUN (channel 1, neuronal marker) and
anti-GFAP (channel 2, glial marker) channels. Averages this per annotated tissue-region
('cluster' obs column) and reports the region with the highest average signal for
each marker.
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

    # channel 0 = DAPI, channel 1 = anti-NEUN (neurons), channel 2 = anti-GFAP (glial cells)
    neun = feat.groupby("cluster")["segmentation_ch-1_mean_intensity_mean"].mean().sort_values()
    gfap = feat.groupby("cluster")["segmentation_ch-2_mean_intensity_mean"].mean().sort_values()

    print("anti-NEUN (neuronal marker) per cluster:")
    print(neun)
    print("train answer (highest avg anti-NEUN signal):", neun.idxmax())

    print("anti-GFAP (glial marker) per cluster:")
    print(gfap)
    print("test answer (highest avg anti-GFAP signal):", gfap.idxmax())


if __name__ == "__main__":
    main()
