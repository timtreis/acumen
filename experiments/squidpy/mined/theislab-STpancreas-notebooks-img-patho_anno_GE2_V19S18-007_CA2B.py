# mined from: https://github.com/theislab/STpancreas/blob/6f9ca0d5ab38422d24db00e8fb4925a856bb6026/notebooks/img/patho_anno_GE2_V19S18-007_CA2B.ipynb
# symbols: squidpy.datasets.visium_fluo_image_crop, squidpy.im.ImageContainer

# %%
import os
import pathlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq

from matplotlib import rcParams
from matplotlib import colors
from seaborn import heatmap
from matplotlib import gridspec

# %load_ext autoreload
# %autoreload 2
# %load_ext lab_black

import PIL

PIL.Image.MAX_IMAGE_PIXELS = 1029959493

# %%
img = sq.im.ImageContainer(
    "/Volumes/external-HDD/dat/SP01-2020_GE_HE_JSON/GE2_V19S18-007_CA2B.jpg",
    library_id="spaceranger",
)

# %%
img.data["image"]

# %%
metadata = pd.read_csv("/Volumes/external-HDD/res/metadata.csv.gz")

# %%
metadata

# %%
adata = sc.read("/Volumes/external-HDD/res/GE2_V19S18-007_CA2Ba.h5ad")

# %%
import imageio
imcont = sq.datasets.visium_fluo_image_crop()
img_1 = imageio.imread(
    "/Users/giovanni.palla/Desktop/Screenshot 2021-07-13 at 16.37.16.png"
)
from skimage.transform import rescale, resize, downscale_local_mean
image_resized = resize(img_1, imcont.shape, anti_aliasing=True)
img.add_img(image_resized, layers="resized")

# %%
#%gui qt

# %%
viewer = img.interactive(adata)

# %%
adata = viewer.adata

# %%
sc.pl.spatial(adata, color="Carcinoma_shapes")

# %%
adata.obs.to_csv("~/Desktop/GE2_V19S18-007_CA2B.csv.gz", index=False)

# %%

