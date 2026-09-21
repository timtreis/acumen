import numpy as np
import squidpy as sq

# --- train variant: fluorescence image, DAPI (nuclear) channel ---
img = sq.datasets.visium_fluo_image_crop()
crop = img.crop_corner(1000, 1000, size=1000)
sq.im.segment(img=crop, layer="image", channel=0)
labels = crop["segmented_watershed"].values
u = np.unique(labels)
n_train = len(u) - (1 if 0 in u else 0)
print("train (visium_fluo, DAPI channel) nuclei count:", n_train)

# --- test variant: H&E image, red channel ---
img2 = sq.datasets.visium_hne_image_crop()
crop2 = img2.crop_corner(0, 0, size=1000)
sq.im.segment(img=crop2, layer="image", channel=0)
labels2 = crop2["segmented_watershed"].values
u2 = np.unique(labels2)
n_test = len(u2) - (1 if 0 in u2 else 0)
print("test (visium_hne, red channel) nuclei count:", n_test)
