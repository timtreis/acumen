# Reproduces the answers for task id: spot_nucleus_count_cluster
#
# Mirrors the analysis in the mined script (03_nn_run/07_dlpfc_squidpy_segment.py):
# smooth the histology image, segment nuclei with watershed, use
# squidpy.im.calculate_image_features(..., features="segmentation") to get a
# per-spot nucleus count, then compare that count across annotated clusters.
#
# train variant -> visium_hne example data
# test variant  -> visium_fluo example data

import squidpy as sq


def run(image_fn, adata_fn, channel=0):
    img = image_fn()
    adata = adata_fn()

    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=channel)

    features_kwargs = {
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label", "centroid"],
            "channels": [channel],
        }
    }

    sq.im.calculate_image_features(
        adata,
        img,
        layer="image",
        key_added="image_features",
        features_kwargs=features_kwargs,
        features="segmentation",
        mask_circle=True,
        n_jobs=1,
    )

    adata.obs["cell_count"] = adata.obsm["image_features"]["segmentation_label"]
    means = adata.obs.groupby("cluster")["cell_count"].mean().sort_values(ascending=False)
    return means


if __name__ == "__main__":
    print("=== train: visium_hne ===")
    means_hne = run(sq.datasets.visium_hne_image_crop, sq.datasets.visium_hne_adata_crop)
    print(means_hne)
    print("TRAIN ANSWER:", means_hne.index[0])

    print("=== test: visium_fluo ===")
    means_fluo = run(sq.datasets.visium_fluo_image_crop, sq.datasets.visium_fluo_adata_crop)
    print(means_fluo)
    print("TEST ANSWER:", means_fluo.index[0])
