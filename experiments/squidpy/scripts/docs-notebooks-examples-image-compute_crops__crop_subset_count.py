import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

# train: crop a square starting at pixel corner (1000, 1000) with side length 800
train_crop = img.crop_corner(1000, 1000, size=800)
train_sub = train_crop.subset(adata)
print("train answer:", train_sub.n_obs)

# test: crop a square starting at pixel corner (3000, 3000) with side length 1500
test_crop = img.crop_corner(3000, 3000, size=1500)
test_sub = test_crop.subset(adata)
print("test answer:", test_sub.n_obs)
