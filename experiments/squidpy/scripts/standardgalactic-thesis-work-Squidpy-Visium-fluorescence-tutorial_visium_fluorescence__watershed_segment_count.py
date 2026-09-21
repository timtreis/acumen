import numpy as np
import squidpy as sq


def count_objects(img):
    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0)
    seg = img["segmented_watershed"].values
    return len(np.unique(seg)) - 1  # exclude background label 0


# train: fluorescence crop
n_fluo = count_objects(sq.datasets.visium_fluo_image_crop())
print("n_objects (fluo):", n_fluo)
print("TRAIN ANSWER (rounded to nearest 5000):", round(n_fluo / 5000) * 5000)

# test: H&E crop
n_hne = count_objects(sq.datasets.visium_hne_image_crop())
print("n_objects (hne):", n_hne)
print("TEST ANSWER (rounded to nearest 5000):", round(n_hne / 5000) * 5000)
