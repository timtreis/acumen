# mined from: https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/9b94a3cde1d03e30639d2469d3d8672749898a8a/Skills/Genomics/Single_Cell/scverse_Ecosystem/squidpy/tutorials/repo/examples/image/compute_gray.ipynb
# symbols: squidpy.datasets.visium_hne_image_crop, squidpy.im.process

# %%
# %matplotlib inline

# %%
import matplotlib.pyplot as plt

import squidpy as sq

# %%
img = sq.datasets.visium_hne_image_crop()

# %%
sq.im.process(img, layer="image", method="gray")

fig, axes = plt.subplots(1, 2)
img.show("image", ax=axes[0])
_ = axes[0].set_title("original")
img.show("image_gray", cmap="gray", ax=axes[1])
_ = axes[1].set_title("grayscale")
