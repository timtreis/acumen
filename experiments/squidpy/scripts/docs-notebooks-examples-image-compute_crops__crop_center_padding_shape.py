import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()

# train: crop centered near the bottom-right edge of the image, radius extends past the boundary
train_crop = img.crop_center(7100, 7100, radius=400)
h, w = train_crop.shape
print("train answer:", f"{h}x{w}")

# test: crop centered near the top-left edge of the image, radius extends past the boundary
test_crop = img.crop_center(100, 150, radius=300)
h, w = test_crop.shape
print("test answer:", f"{h}x{w}")
