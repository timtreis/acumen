# mined from: https://github.com/theislab/segmentdeconv/blob/ee10fc0a421443cd55e0f36776ab8431a0b83902/notebooks/dbitx/6_pp_spatial_data_exploration.py
# symbols: squidpy.im.ImageContainer

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

# %%
import sys
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from stcti.paths import PROJECT_DIR

# %%
# %load_ext autoreload
# %autoreload 2

# %%
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

# %%
pd.set_option("display.max_columns", 500)

# %%
DATA_DIR = PROJECT_DIR / "datasets" / "dbitx"

# %%
list(DATA_DIR.iterdir())

# %%
adata = sc.read(DATA_DIR / "37_48_adata_pp_wohires.h5ad")

# %%
adata

# %%
adata.obs

# %%
image = sq.im.ImageContainer.load(DATA_DIR / "ImageContainers" / "37_48_A1")

# %%
image

# %%
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)

# %%
sc.tl.umap(adata)

# %%
sc.pl.umap(adata[adata.obs.id == "37_48_A1"], color="leiden")

# %%
sc.pl.umap(adata, color="id")

# %%
# adata[adata.obs.id=='37_48_A1'].obs

# %%
z_image = sq.im.ImageContainer()


new_img = np.zeros((21936, 21936, 1, 4))
i = 0
for layer in image:
    if layer == "bf":
        continue
    new_img[..., i] = image[layer].values[..., 0]
    i += 1

z_image.add_img(new_img, layer="37_48_A1")

# %%
z_image.add_img(np.array(new_img, dtype=np.uint8), layer="37_48_A1")

# %%
image = z_image
image

# %%
rows, cols = 5, 5
fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

i = 0
for img, obs in image.generate_spot_crops(
    adata[(adata.obs.id == "37_48_A1") & (adata.obs["leiden"] == "12")],
    library_id="37_48_A1-488",
    return_obs=True,
    spot_diameter_key="spot_diameter_real",
):
    ax[i // cols, i % cols].imshow(img["37_48_A1"].squeeze()[..., :3])
    i += 1

for j in range(i, rows * cols):
    ax[j // cols, j % cols].set_visible(False)

# %%
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(cols * 9, rows * 9))
sc.pl.spatial(
    adata[adata.obs.id == "37_48_A1"], size=5, library_id="37_48_A1-488", ax=ax[0], show=False
)  # , color='leiden')
sc.pl.spatial(
    adata[adata.obs.id == "37_48_A1"],
    size=5,
    library_id="37_48_A1-488",
    ax=ax[1],
    show=False,
    color="leiden",
    groups=["5"],
)
sc.pl.spatial(
    adata[adata.obs.id == "37_48_A1"], size=5, library_id="37_48_A1-488", ax=ax[2], show=False, color="leiden"
)

# %%
rows, cols = 1, 3
fig, ax = plt.subplots(rows, cols, figsize=(cols * 9, rows * 9))
sc.pl.spatial(adata[adata.obs.id == "37_48_A1"], size=5, library_id="37_48_A1-dapi", ax=ax[0], show=False)
sc.pl.spatial(
    adata[adata.obs.id == "37_48_A1"], size=5, library_id="37_48_A1-dapi", ax=ax[1], show=False, color="dapi_mean"
)
sc.pl.spatial(
    adata[adata.obs.id == "37_48_A1"],
    size=5,
    library_id="37_48_A1-488",
    ax=ax[2],
    show=False,
    color="leiden",
    groups=["5"],
)

# %%
adata[adata.obs.id == "37_48_A1"].obs

# %%
