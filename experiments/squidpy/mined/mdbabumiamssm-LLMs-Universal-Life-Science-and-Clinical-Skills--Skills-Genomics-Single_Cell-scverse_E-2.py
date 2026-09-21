# mined from: https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/9b94a3cde1d03e30639d2469d3d8672749898a8a/Skills/Genomics/Single_Cell/scverse_Ecosystem/squidpy/tutorials/repo/examples/image/compute_smooth.ipynb
# symbols: squidpy.datasets.visium_hne_image_crop, squidpy.im.process

# %%
# %matplotlib inline

# %%
import matplotlib.pyplot as plt

import squidpy as sq

# load the H&E stained tissue image
img = sq.datasets.visium_hne_image_crop()

# %%
sq.im.process(img, layer="image", method="smooth", sigma=2)

# %%
crop = img.crop_corner(0, 0, size=200)

fig, axes = plt.subplots(1, 2)
for i, layer in enumerate(["image", "image_smooth"]):
    crop.show(layer, ax=axes[i])
    axes[i].set_title(layer)
