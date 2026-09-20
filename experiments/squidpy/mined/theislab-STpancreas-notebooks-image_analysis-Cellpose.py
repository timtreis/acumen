# mined from: https://github.com/theislab/STpancreas/blob/6f9ca0d5ab38422d24db00e8fb4925a856bb6026/notebooks/image_analysis/Cellpose.ipynb
# symbols: squidpy.im.ImageContainer, squidpy.im.segment

# %%
import squidpy as sq
import numpy as np
import matplotlib.pyplot as plt
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 1029959493

# %%
from cellpose import models

# %%
img = sq.im.ImageContainer(
    "/Volumes/external-HDD/STpancreas/dat/SP01-2020_GE_HE_JSON/GE1_V19S18-084_CA1A.jpg",
    library_id="spaceranger",
)

# %%
def cellpose_he(img, min_size=15, flow_threshold=0.5, channel_cellpose=0):
    model = models.Cellpose(model_type='nuclei')
    res, _, _, _ = model.eval(
        img,
        channels=[channel_cellpose, 0],
        diameter=None,
        min_size=min_size,
        invert=True,
        flow_threshold=flow_threshold,
    )
    return res

# %%
for i in np.arange(0.5,1.6,0.1):
    sq.im.segment(img=img, layer="image", channel=None, method=cellpose_he, channel_cellpose=0,flow_threshold=i,layer_added=f"Flow_threshold_{i}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 20))
img.show("image", channel=None, ax=axes[0])
_ = axes[0].set_title("H&E")
img.show("Flow_threshold", cmap="jet", interpolation="none", ax=axes[1]) #choose the value of flow threshold to plot from the grid search
_ = axes[1].set_title("Cellpose segmentation")

# %%

