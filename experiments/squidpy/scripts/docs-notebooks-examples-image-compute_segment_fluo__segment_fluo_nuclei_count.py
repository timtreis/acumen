"""
Ground-truth script for task `segment_fluo_nuclei_count`.

Reproduces squidpy's docs/notebooks/examples/image/compute_segment_fluo.ipynb:
load the visium_fluo_image_crop dataset, crop a 1000x1000 pixel region, run
watershed segmentation (squidpy.im.segment) on channel 0 (DAPI) with automatic
(Otsu) thresholding, and count the number of unique labels in the resulting
segmentation mask (including the background label 0) -- exactly what the
notebook itself prints as "Number of segments in crop".
"""

import numpy as np
import squidpy as sq


def run(y, x, size=1000):
    img = sq.datasets.visium_fluo_image_crop()
    crop = img.crop_corner(y, x, size=size)
    sq.im.segment(img=crop, layer="image", channel=0, method="watershed", thresh=None, geq=True)
    n_labels = len(np.unique(crop["segmented_watershed"]))
    return n_labels


if __name__ == "__main__":
    train_n = run(1000, 1000, size=1000)
    test_n = run(3000, 2000, size=1000)
    print("train (region at row=1000, col=1000, size=1000):", train_n)
    print("test  (region at row=3000, col=2000, size=1000):", test_n)
