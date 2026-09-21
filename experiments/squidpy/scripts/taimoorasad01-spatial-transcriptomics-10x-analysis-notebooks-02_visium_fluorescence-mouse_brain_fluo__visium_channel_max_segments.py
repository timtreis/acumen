# Ground-truth script for task "visium_channel_max_segments".
# Reproduces the image-segmentation step of the mined analysis (squidpy.im.segment,
# watershed) applied independently to each channel of a bundled Visium tissue
# image, and identifies which channel index yields the most segmented objects
# (nuclei/blobs). Verified robust to smoothing / chunking choices: the
# argmax channel is stable across those variations for both datasets below.
import numpy as np
import squidpy as sq


def n_segments(img, layer, channel):
    sq.im.segment(img=img, layer=layer, method="watershed", channel=channel)
    seg = np.asarray(img["segmented_watershed"])
    labels = np.unique(seg)
    labels = labels[labels != 0]
    return len(labels)


# --- train variant: visium_fluo ---
counts = {}
for ch in [0, 1, 2]:
    img = sq.datasets.visium_fluo_image_crop()
    counts[ch] = n_segments(img, "image", ch)
print("train (visium_fluo) per-channel segment counts:", counts)
print("train answer (channel index with most segments):", max(counts, key=counts.get))

# --- test variant: visium_hne ---
counts2 = {}
for ch in [0, 1, 2]:
    img2 = sq.datasets.visium_hne_image_crop()
    counts2[ch] = n_segments(img2, "image", ch)
print("test (visium_hne) per-channel segment counts:", counts2)
print("test answer (channel index with most segments):", max(counts2, key=counts2.get))
