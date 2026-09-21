"""Ground truth for task 'nuclei_max_spot' (train + test variants).

Both variants segment the cell nuclei in the fluorescent Visium brain
tissue crop bundled with the package.

Train: which spot (barcode) contains the most nuclei.
Test: how many nuclei are detected across the whole cropped image.
"""

import numpy as np
import squidpy as sq


def main():
    img = sq.datasets.visium_fluo_image_crop()
    adata = sq.datasets.visium_fluo_adata_crop()

    sq.im.segment(
        img=img,
        layer="image",
        layer_added="segmented_watershed",
        method="watershed",
        channel=0,
    )

    # --- train: spot with the most nuclei ---
    sq.im.calculate_image_features(
        adata,
        img,
        layer="image",
        features="segmentation",
        key_added="segmentation_features",
        features_kwargs={
            "segmentation": {
                "label_layer": "segmented_watershed",
                "props": ["label"],
            }
        },
        mask_circle=True,
        n_jobs=1,
    )

    df = adata.obsm["segmentation_features"]
    top_barcode = df["segmentation_label"].idxmax()
    print("[train] Spot with most nuclei:", top_barcode)
    print("[train] Nucleus count at that spot:", df["segmentation_label"].max())

    # --- test: total nuclei across the whole cropped image ---
    labels = np.unique(img["segmented_watershed"].values)
    total_nuclei = len(labels) - 1  # exclude background label 0
    print("[test] Total nuclei detected in image:", total_nuclei)


if __name__ == "__main__":
    main()
