# mined from: https://github.com/NBISweden/single-cell_sib_scilifelab_2021/blob/f88496f6af4f6780f1769768f812583bc790ad77/project_spatial/notebooks/napari-test.ipynb
# symbols: squidpy.im.ImageContainer

# %%
import scanpy as sc
import squidpy as sq
import numpy as np
import os.path as osp
from PIL import Image

# %%
# image data file
IMG_PTH = "../../../data/spatial/hippo/rawdata/ST8059049/spatial/tissue_hires_image.png"
# data directory
DATA_DIR = osp.dirname(osp.dirname(IMG_PTH))
# new library name (we will change to this)
new_library_id = osp.basename(DATA_DIR)

# %%
# read the spatial data
adata = sc.read_visium(DATA_DIR)
# load image and convert to numpy array, we also transpose the image
img = np.array(Image.open(IMG_PTH)).transpose(2,0,1)

# %%
# get old library name
old_library_id = list(adata.uns["spatial"].keys())[0]
old_library_id

# %%
adata.uns["spatial"][new_library_id] = adata.uns["spatial"].pop(old_library_id)

# %%
scalefactor = adata.uns["spatial"][new_library_id]["scalefactors"]["tissue_hires_scalef"]

# %%
img = sq.im.ImageContainer(img, # image object
                           dims=("channels", "y", "x"), # name of dimensions
                           layer="img1", # layer name
                           scale = scalefactor, # scalefactor to use
                           library_id = new_library_id, # library name
                          )

# %%
viewer = img.interactive(adata)

# %%
adata.obs = viewer.adata.obs
adata

# %%
sc.pl.spatial(adata,color = ["hippocampus_shapes"])
