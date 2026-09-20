# mined from: https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/9b94a3cde1d03e30639d2469d3d8672749898a8a/Skills/Genomics/Single_Cell/scverse_Ecosystem/squidpy/tutorials/repo/tutorials/tutorial_image_container.ipynb
# symbols: squidpy.datasets.visium_hne_image, squidpy.im.ImageContainer

# %%
# %matplotlib inline

# %%
import numpy as np

import squidpy as sq

# %%
arr = np.ones((100, 100, 3))
arr[40:60, 40:60] = [0, 0.7, 1]

print(arr.shape)
img = sq.im.ImageContainer(arr, layer="img1")
img

# %%
arr1 = arr.transpose(2, 0, 1)
print(arr1.shape)
img = sq.im.ImageContainer(arr1, dims=("channels", "y", "x"), layer="img1")
img

# %%
arr_seg = np.zeros((100, 100))
arr_seg[40:60, 40:60] = 1

img.add_img(arr_seg, layer="seg1")
img

# %%
img["seg2"] = arr_seg
img

# %%
print(list(img))
img["img1"]

# %%
img.rename("seg2", "new-name")

# %%
img.show(layer="img1")

# %%
crop1 = img.crop_corner(30, 40, size=(30, 30), scale=1)
crop1.show(layer="img1")

crop2 = crop1.crop_corner(0, 0, size=(40, 40), scale=0.5)
crop2.show(layer="img1")

# %%
print(crop1.data.attrs)
print(crop2.data.attrs)

sq.im.ImageContainer.uncrop([crop1], shape=img.shape).show(layer="img1")
sq.im.ImageContainer.uncrop([crop2], shape=(50, 50)).show(layer="img1")

# %%
img.data

# %%
img_on_disk = sq.datasets.visium_hne_image()
print(type(img_on_disk["image"].data))

# %%
img_on_disk.compute()
print(type(img_on_disk["image"].data))
