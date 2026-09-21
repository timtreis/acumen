# mined from: https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/9b94a3cde1d03e30639d2469d3d8672749898a8a/Skills/Genomics/Single_Cell/scverse_Ecosystem/squidpy/tutorials/repo/examples/image/compute_process_hires.ipynb
# symbols: squidpy.datasets.visium_hne_image, squidpy.im.ImageContainer, squidpy.im.process

# %%
# %matplotlib inline

# %%
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

import squidpy as sq

# %%
# load the H&E stained tissue image
img = sq.datasets.visium_hne_image()

# %%
sq.im.process(img, layer="image", method="gray", chunks=1000)

# %%
crop = img.crop_corner(4000, 4000, size=2000)

fig, axes = plt.subplots(1, 2)
crop.show("image", ax=axes[0])
_ = axes[0].set_title("original")
crop.show("image_gray", cmap="gray", ax=axes[1])
_ = axes[1].set_title("grayscale")

# %%
arr = np.zeros((20, 20))
arr[10:] = 1
img = sq.im.ImageContainer(arr, layer="image")

# smooth the image using `depth` 0 and 1
sq.im.process(
    img,
    layer="image",
    method=gaussian_filter,
    layer_added="smooth_depth0",
    chunks=10,
    sigma=1,
    apply_kwargs={"depth": 0},
)
sq.im.process(
    img,
    layer="image",
    method=gaussian_filter,
    layer_added="smooth_depth1",
    chunks=10,
    sigma=1,
    apply_kwargs={"depth": 1, "boundary": "reflect"},
)

# %%
fig, axes = plt.subplots(1, 3)
img.show("image", ax=axes[0])
_ = axes[0].set_title("original")
img.show("smooth_depth0", ax=axes[1])
_ = axes[1].set_title("non-overlapping crops")
img.show("smooth_depth1", ax=axes[2])
_ = axes[2].set_title("overlapping crops")
