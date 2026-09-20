# mined from: https://github.com/theislab/stcti_notebook/blob/92f8c5497a6a6e3b6f3749c9905f5782b0244fed/image_preprocessing/pp_compute_masks.ipynb
# symbols: squidpy.im.ImageContainer, squidpy.im.segment

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import torch 

from cellpose import models
from stcti.paths import DATA_DIR, PROJECT_DIR
from PIL import Image

# %%
# adata_in = DATA_DIR / 'dbitx_kidney' / '37_38_kidney_preprocessed_with_hq_images.h5ad'
# adata = sc.read(adata_in)

# %%
# image_lowres = adata.uns['spatial']['B3_dapi']['images']['lowres']
# image = adata.uns['spatial']['B3_dapi']['images']['hires']

# %%
import os

file_list = [file_name for file_name in os.listdir(PROJECT_DIR/"37_44") if 's7' in file_name]
image_paths = [PROJECT_DIR/"37_44"/f"37_44_13x13_10%_Shading_Correction-01_s7c{i+1}_ORG.tif" for i in range(3)]

image_paths

# %%
# im = Image.fromarray(image_path)
# im

# %%
Image.MAX_IMAGE_PIXELS = 413349561
im = Image.open(image_paths[0])

im.mode

# %%
image = np.array([np.array(Image.open(fp), dtype=np.uint16) for fp in image_paths])

# %%
image = sq.im.ImageContainer(image, lazy=True, chunks=200)

# %%
# com_15626_16459.tif
crop = image.crop_center(15626, 16459, radius=132)

fig, ax = plt.subplots(1, 4, figsize=(3*5,5))
crop.show(channelwise=True, ax=ax[:-1], layer='image')
crop.show(channelwise=True, layer='segmented_custom', ax=ax[-1])

# %%
image

# %%
gpu = False
if torch.cuda.is_available():
    gpu = True

print(f'gpu: {gpu}')

# %%
crop

# %%
def cellpose(img, min_size=15):
    model = models.Cellpose(model_type='nuclei', gpu=gpu)
    res, _, _, _ = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        min_size=min_size,
    )
    return res

# %%
sq.im.segment(img=image, layer="image", channel=0, method=cellpose)

# %%
crop = image.crop_corner(10000, 13000, size=600)

# compute shift to distinguish between segments 
shift = np.unique(crop['segmented_custom'])[1]
crop_array = np.where(crop['segmented_custom']==0, 0, crop['segmented_custom']-shift).squeeze()

fig, ax = plt.subplots(1, 2, figsize=(10, 20))
crop.show("image", channel=0, ax=ax[0])
ax[0].set_title("DAPI")
ax[1].imshow(crop_array, cmap="jet", interpolation="none")
ax[1].set_title("Cellpose segmentation")
ax[1].axis('off')

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

# %%
image

# %%
file_path = PROJECT_DIR / 'brain_images'/ 'segmentation'
image.save(file_path)

# %%
_image = sq.im.ImageContainer().load(file_path)

# %%
_image

# %%

