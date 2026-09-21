"""
Ground-truth script for task `stardist_nuclei_segment_count`.

The real tutorial (tutorial_stardist.ipynb) segments nuclei in crops of the
visium_fluo_image_crop / visium_hne_image_crop datasets using a StarDist
pretrained deep-learning model plugged into squidpy.im.segment as a custom
callable, then counts the resulting number of segments (np.unique of the
label image, minus background).

`stardist`/`csbdeep`/`tensorflow` are not available in this environment and
may not be installed here, so the actual StarDist model cannot be run. To
still exercise the same squidpy segmentation entry point
(`squidpy.im.segment`) on the same bundled dataset and reproduce the same
kind of "segment nuclei, count objects" analysis deterministically, this
script uses squidpy's built-in default segmentation backend (Otsu threshold +
watershed) instead of a custom StarDist callable, applied directly to each
half of the image (no extra preprocessing), which is the simplest, most
direct way a user of the package would carry out the same goal.
"""

import numpy as np
import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
height, width = img["image"].shape[0], img["image"].shape[1]
half = width // 2

# train: left half of the image
left = img.crop_corner(0, 0, size=(height, half))
sq.im.segment(img=left, layer="image", channel=0, layer_added="segmented")
n_left = len(np.unique(left["segmented"].values)) - 1  # exclude background label 0
print("train (left half) n_segments raw =", n_left)
print("train answer (rounded to nearest 500) =", round(n_left / 500) * 500)

# test: right half of the image
right = img.crop_corner(0, half, size=(height, width - half))
sq.im.segment(img=right, layer="image", channel=0, layer_added="segmented")
n_right = len(np.unique(right["segmented"].values)) - 1
print("test (right half) n_segments raw =", n_right)
print("test answer (rounded to nearest 500) =", round(n_right / 500) * 500)
