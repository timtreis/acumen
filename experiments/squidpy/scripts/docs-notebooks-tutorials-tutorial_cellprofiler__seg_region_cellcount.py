"""
Ground truth for task `seg_region_cellcount`.

Mirrors the squidpy CellProfiler tutorial's own pipeline (docs/notebooks/tutorials/tutorial_cellprofiler.ipynb):
  1. Load the visium_fluo image + matching AnnData.
  2. Smooth + watershed-segment cells in the fluorescence image (squidpy.im.process / squidpy.im.segment).
  3. For each Visium spot, crop the segmented image and count the number of segmented
     cell objects that fall inside it (squidpy.im.calculate_image_features, features='segmentation').
  4. Average that per-spot cell count within each annotated anatomical region (adata.obs['cluster'])
     and report the region with the highest / lowest average.

Verified robust to choice of smoothing sigma (None, 1, 2, 3) and segmentation channel (0, 1, 2):
highest average count -> Hypothalamus_1, lowest average count -> Cortex_2 in all cases tested.
"""

import pandas as pd
import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

sq.im.process(img=img, layer="image", method="smooth", sigma=[2, 2, 0, 0])
sq.im.segment(img=img, layer="image_smooth", method="watershed", channel_ids=0, chunks=1000)

df = sq.im.calculate_image_features(
    adata,
    img,
    layer="image_smooth",
    features="segmentation",
    features_kwargs={
        "segmentation": {"label_layer": "segmented_watershed", "props": ["label"]}
    },
    copy=True,
    show_progress_bar=False,
)

df = df.loc[adata.obs_names]
df["region"] = adata.obs["cluster"].values

avg_count_per_region = df.groupby("region")["segmentation_label"].mean().sort_values()

print(avg_count_per_region)
print()
print("TRAIN answer (highest average segmented-cell count per spot):", avg_count_per_region.idxmax())
print("TEST answer (lowest average segmented-cell count per spot):", avg_count_per_region.idxmin())
