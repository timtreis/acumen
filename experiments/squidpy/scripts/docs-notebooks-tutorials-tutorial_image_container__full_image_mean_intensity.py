"""Ground-truth script for task 'full_image_mean_intensity'.

Loads a bundled spatial image dataset fully into memory (forcing computation
of the lazily-backed on-disk array) and reports the mean pixel intensity
across the whole image (all color channels).
"""
import squidpy as sq

# --- train variant: full-resolution, uncropped Visium H&E mouse-brain image ---
img = sq.datasets.visium_hne_image()
img.compute()
train_mean = img["image"].values.mean()
print("train answer:", round(float(train_mean), 2))

# --- test variant: Visium fluorescence image ---
img_fluo = sq.datasets.visium_fluo_image_crop()
img_fluo.compute()
test_mean = img_fluo["image"].values.mean()
print("test answer:", round(float(test_mean), 2))
