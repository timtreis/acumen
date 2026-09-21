"""Confirmation script for task: nuclei_segmentation_count

Reproduces the train and test answers by segmenting cell nuclei in a bundled
squidpy tissue image (Gaussian smoothing followed by watershed segmentation
on channel 0) and counting the number of distinct segmented objects.
"""
import numpy as np
import squidpy as sq


def count_nuclei(dataset_fn):
    img = dataset_fn()
    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0)
    seg = img["segmented_watershed"].values
    uniq = np.unique(seg)
    n_labels = len(uniq) - (1 if uniq.min() == 0 else 0)
    return n_labels


if __name__ == "__main__":
    train_count = count_nuclei(sq.datasets.visium_fluo_image_crop)
    print("train (visium_fluo_image_crop) nuclei count:", train_count)

    test_count = count_nuclei(sq.datasets.visium_hne_image_crop)
    print("test (visium_hne_image_crop) nuclei count:", test_count)
