import squidpy as sq

# TRAIN variant: visium_hne data
adata = sq.datasets.visium_hne_adata()
img = sq.datasets.visium_hne_image()
gen = img.generate_spot_crops(
    adata,
    obs_names=[adata.obs_names[0]],
    spot_scale=1.5,
    as_array="image",
    return_obs=False,
)
crop = next(gen)
print("TRAIN (visium_hne) crop shape:", crop.shape)  # -> (133, 133, 3)

# TEST variant: visium_fluo_crop data
adata2 = sq.datasets.visium_fluo_adata_crop()
img2 = sq.datasets.visium_fluo_image_crop()
gen2 = img2.generate_spot_crops(
    adata2,
    obs_names=[adata2.obs_names[0]],
    spot_scale=1.5,
    as_array="image",
    return_obs=False,
)
crop2 = next(gen2)
print("TEST (visium_fluo_crop) crop shape:", crop2.shape)  # -> (269, 269, 3)
