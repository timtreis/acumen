"""Ground-truth script for task `custom_mean_feature`.

Demonstrates the notebook's core analysis: defining a custom python
function (mean pixel intensity) and using it as a squidpy image feature,
computed per spot for a Visium crop dataset.
"""

import numpy as np
import squidpy as sq


def mean_fn(arr):
    """Compute mean of arr."""
    return np.mean(arr)


def top_spot(img, adata):
    sq.im.calculate_image_features(
        adata,
        img,
        features="custom",
        features_kwargs={"custom": {"func": mean_fn}},
        key_added="custom_features",
        show_progress_bar=False,
    )
    df = adata.obsm["custom_features"]
    top = df["mean_fn_0"].idxmax()
    return top, df["mean_fn_0"].max()


if __name__ == "__main__":
    # train: cropped Visium H&E dataset
    img_hne = sq.datasets.visium_hne_image_crop()
    adata_hne = sq.datasets.visium_hne_adata_crop()
    top_hne, val_hne = top_spot(img_hne, adata_hne)
    print("TRAIN (H&E) top spot:", top_hne, val_hne)

    # test: cropped Visium fluorescence dataset
    img_fluo = sq.datasets.visium_fluo_image_crop()
    adata_fluo = sq.datasets.visium_fluo_adata_crop()
    top_fluo, val_fluo = top_spot(img_fluo, adata_fluo)
    print("TEST (fluorescence) top spot:", top_fluo, val_fluo)
