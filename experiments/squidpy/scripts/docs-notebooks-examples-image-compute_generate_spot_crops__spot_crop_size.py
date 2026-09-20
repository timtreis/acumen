"""Confirms answers for the 'spot_crop_size' task (train=visium_hne, test=visium_fluo)."""
import squidpy as sq

# --- train: visium_hne ---
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()
gen = img.generate_spot_crops(adata, spot_scale=2.0, as_array="image", squeeze=True)
crop = next(gen)
print("train (visium_hne) first crop shape:", crop.shape)  # -> (177, 177, 3), answer: 177

# --- test: visium_fluo ---
img2 = sq.datasets.visium_fluo_image_crop()
adata2 = sq.datasets.visium_fluo_adata_crop()
gen2 = img2.generate_spot_crops(adata2, spot_scale=2.0, as_array="image", squeeze=True)
crop2 = next(gen2)
print("test (visium_fluo) first crop shape:", crop2.shape)  # -> (357, 357, 3), answer: 357
