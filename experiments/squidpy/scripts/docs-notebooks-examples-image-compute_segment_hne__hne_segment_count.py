"""Ground-truth script for task `hne_segment_count` (train + test variants).

Reproduces the H&E cell-segmentation workflow from
docs/notebooks/examples/image/compute_segment_hne.ipynb:
smooth the H&E crop, then run watershed segmentation on channel 0 with a
manual intensity threshold of 90, treating darker pixels as foreground
(geq=False) since nuclei stain dark in H&E (opposite of fluorescence).
"""

import numpy as np
import squidpy as sq


def segment_count(corner_y, corner_x, size=1000):
    img = sq.datasets.visium_hne_image_crop()
    crop = img.crop_corner(corner_y, corner_x, size=size)
    sq.im.process(crop, layer="image", method="smooth", sigma=4)
    sq.im.segment(img=crop, layer="image_smooth", method="watershed", thresh=90, geq=False)
    labels = crop["segmented_watershed"].values
    return len(np.unique(labels))


if __name__ == "__main__":
    train_answer = segment_count(0, 0, size=1000)
    print("train (top-left 1000x1000 crop):", train_answer)

    test_answer = segment_count(1000, 1000, size=1000)
    print("test (1000,1000 1000x1000 crop):", test_answer)
