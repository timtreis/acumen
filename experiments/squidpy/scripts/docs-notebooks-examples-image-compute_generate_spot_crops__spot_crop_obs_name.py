"""Confirms answers for the 'spot_crop_obs_name' task (train=visium_hne, test=visium_fluo)."""
import squidpy as sq

# --- train: visium_hne ---
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()
gen = img.generate_spot_crops(adata, as_array="image", squeeze=True, return_obs=True)
for _ in range(5):
    image, obs_name = next(gen)
print("train (visium_hne) 5th crop obs_name:", obs_name)  # answer: AAATGGTCAATGTGCC-1

# --- test: visium_fluo ---
img2 = sq.datasets.visium_fluo_image_crop()
adata2 = sq.datasets.visium_fluo_adata_crop()
gen2 = img2.generate_spot_crops(adata2, as_array="image", squeeze=True, return_obs=True)
for _ in range(5):
    image2, obs_name2 = next(gen2)
print("test (visium_fluo) 5th crop obs_name:", obs_name2)  # answer: AAATTAACGGGTAGCT-1
