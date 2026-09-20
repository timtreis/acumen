"""Ground-truth script for task 'crop_region_mean_intensity'.

Crops a defined square pixel region out of a bundled spatial image dataset
and reports the mean pixel intensity across that region (all color channels).
"""
import squidpy as sq

# --- train variant: Visium H&E mouse-brain image ---
img = sq.datasets.visium_hne_image_crop()
crop = img.crop_corner(1000, 1000, size=(500, 500))
train_mean = crop["image"].values.mean()
print("train answer:", round(float(train_mean), 2))

# --- test variant: Visium fluorescence image ---
img_fluo = sq.datasets.visium_fluo_image_crop()
crop_fluo = img_fluo.crop_corner(2000, 2000, size=(500, 500))
test_mean = crop_fluo["image"].values.mean()
print("test answer:", round(float(test_mean), 2))
