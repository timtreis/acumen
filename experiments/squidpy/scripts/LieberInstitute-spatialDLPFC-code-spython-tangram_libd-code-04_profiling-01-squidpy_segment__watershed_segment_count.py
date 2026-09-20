import squidpy as sq
import numpy as np

# train variant: visium_fluo_image_crop
img = sq.datasets.visium_fluo_image_crop()
sq.im.process(img=img, layer="image", method="smooth")
sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0)
seg = img["segmented_watershed"].values
labels = np.unique(seg)
n_nonzero = len(labels) - (1 if 0 in labels else 0)
print("train (visium_fluo_image_crop) segments:", n_nonzero)

# test variant: visium_hne_image_crop
img2 = sq.datasets.visium_hne_image_crop()
sq.im.process(img=img2, layer="image", method="smooth")
sq.im.segment(img=img2, layer="image_smooth", method="watershed", channel=0)
seg2 = img2["segmented_watershed"].values
labels2 = np.unique(seg2)
n_nonzero2 = len(labels2) - (1 if 0 in labels2 else 0)
print("test (visium_hne_image_crop) segments:", n_nonzero2)
