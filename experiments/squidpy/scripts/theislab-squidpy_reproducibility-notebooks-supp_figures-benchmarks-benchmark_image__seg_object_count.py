import squidpy as sq
import numpy as np

# train: visium fluorescence crop dataset
img_train = sq.datasets.visium_fluo_image_crop()
sq.im.segment(img_train)
seg_train = img_train["segmented_watershed"].values
n_train = int(len(np.unique(seg_train)) - 1)  # subtract background label 0
print("train (visium_fluo_crop) object count:", n_train)

# test: visium H&E crop dataset
img_test = sq.datasets.visium_hne_image_crop()
sq.im.segment(img_test)
seg_test = img_test["segmented_watershed"].values
n_test = int(len(np.unique(seg_test)) - 1)
print("test (visium_hne_crop) object count:", n_test)
