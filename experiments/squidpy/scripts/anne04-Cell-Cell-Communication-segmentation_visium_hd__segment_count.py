import numpy as np
import squidpy as sq

# ---- train variant: H&E crop ----
img = sq.datasets.visium_hne_image_crop()
sq.im.process(img, layer="image", method="smooth")
sq.im.segment(img, channel=0, layer="image_smooth", method="watershed", geq=False, layer_added="segmented_watershed")
n_train = len(np.unique(img["segmented_watershed"]))
print("train (visium_hne_image_crop) number of segments:", n_train)

# ---- test variant: full-resolution H&E image ----
img2 = sq.datasets.visium_hne_image()
sq.im.process(img2, layer="image", method="smooth")
sq.im.segment(img2, channel=0, layer="image_smooth", method="watershed", geq=False, layer_added="segmented_watershed")
n_test = len(np.unique(img2["segmented_watershed"]))
print("test (visium_hne_image) number of segments:", n_test)
