# mined from: https://github.com/theislab/segmentdeconv/blob/ee10fc0a421443cd55e0f36776ab8431a0b83902/notebooks/dbitx/whole_silde_image_analysis/2_pp_cell_segmentation.py
# symbols: squidpy.im.ImageContainer, squidpy.im.segment

# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %% [markdown]
# **Requires:**
# - -
#
# **Outputs:**
# - `DATA_DIR /human_brain_V1 /segmentation `
# ---
#
# ## Library imports
#
# from PIL import Image
# from pathlib import Path
# from cellpose import models

# %%
import sys
import torch
import scanpy as sc
import squidpy as sq
import tifffile as tiffio
import matplotlib.pyplot as plt

import numpy as np

from stcti.paths import PROJECT_DIR

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Load image

# %%
storage = PROJECT_DIR / "datasets" / "dbitx"
folder = "220303_kidney_wholesections"

DATA_DIR = storage / folder
file_name = "SBY_04_cond3_wholesection_AQP1+Nephrin.tif"
image_path = DATA_DIR / file_name

print(f"Valid image path: {image_path.exists()}")

# %%
image = sq.im.ImageContainer()

for file in DATA_DIR.iterdir():
    if file.suffix != ".tif":
        continue
    if file.name == file_name:
        channel = file.name
        layer = channel.split(".")[0].split("_")[-1].replace("+", "_")
        img = tiffio.imread(file)
        # rearrange stain1, stain2, DAPI, Phalloidin
        img = img[[4, 0, 3, 1], ...]
        # normalise
        img = img.astype(np.float64) / img.max()  # normalize the data to 0 - 1
        img = 255 * img  # Now scale by 255
        img = img.astype(np.uint8)
        image.add_img(img, layer=layer)


# %%
image

# %% [markdown]
# ## Apply cellpose

# %%
gpu = False
if torch.cuda.is_available():
    gpu = True

print(f"gpu: {gpu}")


# %%
def cellpose(img, min_size=15):
    model = models.Cellpose(model_type="nuclei", gpu=gpu)
    res, _, _, _ = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        min_size=min_size,
    )
    return res


# %%
crop = image.crop_corner(7000, 7000, size=600)
crop.show(channelwise=True, layer="AQP1_Nephrin")
crop.show(channel=[0, 1, 3], layer="AQP1_Nephrin")

# %% tags=[]
sq.im.segment(img=image, layer="AQP1_Nephrin", channel=2, method=cellpose)

# %% [markdown]
# **Plot segmentation in crop of image**

# %%
crop = image.crop_corner(7000, 7000, size=600)

# compute shift to distinguish between segments
shift = np.unique(crop["segmented_custom"])[1]
crop_array = np.where(crop["segmented_custom"] == 0, 0, crop["segmented_custom"] - shift).squeeze()

fig, ax = plt.subplots(1, 2, figsize=(10, 20))
crop.show("AQP1_Nephrin", channel=2, ax=ax[0])
ax[0].set_title("DAPI")
ax[1].imshow(crop_array, cmap="jet", interpolation="none")
ax[1].set_title("Cellpose segmentation")
ax[1].axis("off")

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

# %%
image

# %% [markdown]
# ## Save

# %%
image.save(DATA_DIR / (file_name.split(".")[0] + "_segmented"))

# %% [markdown]
# Check that it works correctly

# %%
_image = sq.im.ImageContainer().load(DATA_DIR / (file_name.split(".")[0] + "_segmented"))

# %%
_image

# %%
