"""
Confirmation script for task: img_seg_fragmented_cluster (train + test variants).

Goal: segment the tissue image into sub-regions based on pixel intensity,
count how many sub-regions fall within each spot, and find which existing
tissue cluster has the highest average count of sub-regions per spot.

train dataset: visium_fluo (bundled squidpy example: fluorescence Visium crop)
test dataset:  visium_hne  (bundled squidpy example: H&E Visium crop)

Uses squidpy's own default parameters for `im.segment` (method="watershed",
channel=0, no chunking) — the same defaults the original analysis relied on.
"""

import squidpy as sq


def top_cluster_by_segment_count(img, adata):
    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed", layer_added="segmented_watershed")

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

    df = adata.obsm["features_segmentation"].copy()
    df["cluster"] = adata.obs["cluster"].values
    mean_per_cluster = df.groupby("cluster")["segmentation_label"].mean().sort_values(ascending=False)
    return mean_per_cluster


if __name__ == "__main__":
    # --- train: visium_fluo ---
    img_fluo = sq.datasets.visium_fluo_image_crop()
    adata_fluo = sq.datasets.visium_fluo_adata_crop()
    ranking_fluo = top_cluster_by_segment_count(img_fluo, adata_fluo)
    print(ranking_fluo)
    print("TRAIN ANSWER:", ranking_fluo.index[0])

    # --- test: visium_hne ---
    img_hne = sq.datasets.visium_hne_image_crop()
    adata_hne = sq.datasets.visium_hne_adata_crop()
    ranking_hne = top_cluster_by_segment_count(img_hne, adata_hne)
    print(ranking_hne)
    print("TEST ANSWER:", ranking_hne.index[0])
