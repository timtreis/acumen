import numpy as np
import squidpy as sq

# --- train ---
img = sq.datasets.visium_hne_image()
sq.im.process(img, layer="image", method="gray", chunks=1000)
crop = img.crop_corner(4000, 4000, size=2000)
gray = np.asarray(crop["image_gray"].values)
print("train mean:", round(float(gray.mean()), 2))

# --- test ---
img2 = sq.datasets.visium_hne_image_crop()
sq.im.process(img2, layer="image", method="gray", chunks=500)
crop2 = img2.crop_corner(1000, 1000, size=1000)
gray2 = np.asarray(crop2["image_gray"].values)
print("test mean:", round(float(gray2.mean()), 2))
