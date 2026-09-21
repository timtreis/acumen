"""Reproduces the answers for task 'spot_std_by_channel'.

Loads the fluorescence Visium crop dataset (image + adata), computes per-spot
summary intensity statistics (quantile/mean/std) restricted to the tissue
under each spot's circular footprint, for the DAPI (channel 0) and GFAP
(channel 1) fluorescence channels, and reports the spot with the highest
standard deviation for each channel.
"""

import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

sq.im.calculate_image_features(
    adata,
    img,
    features="summary",
    features_kwargs={
        "summary": {
            "quantiles": [0.1],
            "channels": [0, 1],
        }
    },
    key_added="summary_features",
    mask_circle=True,
    show_progress_bar=False,
)

df = adata.obsm["summary_features"]

train_answer = df["summary_ch-0_std"].idxmax()  # DAPI channel
test_answer = df["summary_ch-1_std"].idxmax()  # GFAP channel

print("train (DAPI, channel 0) answer:", train_answer, df["summary_ch-0_std"].max())
print("test (GFAP, channel 1) answer:", test_answer, df["summary_ch-1_std"].max())
