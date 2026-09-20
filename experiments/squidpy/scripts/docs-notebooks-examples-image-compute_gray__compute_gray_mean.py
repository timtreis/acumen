import numpy as np
import squidpy as sq

# --- train variant: Visium H&E image crop ---
img_hne = sq.datasets.visium_hne_image_crop()
sq.im.process(img_hne, layer="image", method="gray")
arr_hne = np.asarray(img_hne["image_gray"].data)
train_answer = round(float(arr_hne.mean()), 3)
print("train (visium_hne_image_crop) mean gray:", train_answer)

# --- test variant: Visium fluorescence image crop ---
img_fluo = sq.datasets.visium_fluo_image_crop()
sq.im.process(img_fluo, layer="image", method="gray")
arr_fluo = np.asarray(img_fluo["image_gray"].data)
test_answer = round(float(arr_fluo.mean()), 3)
print("test (visium_fluo_image_crop) mean gray:", test_answer)
