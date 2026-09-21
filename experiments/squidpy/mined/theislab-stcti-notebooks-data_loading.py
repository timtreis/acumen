# mined from: https://github.com/theislab/stcti/blob/9303d1eb946e4abe5140412b7b7d429c15b9c04b/notebooks/data_loading.ipynb
# symbols: squidpy.im.ImageContainer

# %%
import imageio
import os 
from os.path import join, exists 
import matplotlib.pyplot as plt 
import scanpy as sc
import squidpy as sq
from PIL import Image 

from stcti.vae import ImageDataModule

# %%
# %load_ext autoreload
# %autoreload 2

# %%
sq.__file__

# %%
storage = '/storage/groups/ml01/datasets/raw/20200909_PublicVisium_giovanni.palla'
folder = "2020_10XFluoVisium_HumanBrain1_giovanni.palla"
dataset_name = 'V1_Human_Brain_Section_1'


image_path = join(join(storage, folder), dataset_name+'_image.tif')
print(f"Valid image path: {exists(image_path)}")

adata_path = f'../datasets/{dataset_name}.h5ad'
print(f"Valid adata path: {exists(adata_path)}")

# %%
image = sq.im.ImageContainer(image_path, lazy=True, chunks=200)
adata = sc.read(adata_path)

display(image)
print()
display(adata)

# %%
img = adata.uns['spatial']['V1_Human_Brain_Section_1']['images']['hires']
plt.imshow(img)

# %%
from PIL import Image 
import numpy as np 
from tqdm.notebook import tqdm


def write_img_from_gen(gen, dir_name, scale=1.6): 
    if exists(dir_name): 
        print(f'Directory "{dir_name}" already exists.')
    else: 
        os.mkdir(dir_name)

    overwrite = False
    for im, obs in tqdm(gen): 
        fname = f'{dir_name}/{obs}.tif'
        if exists(fname):
            if not overwrite:
                print(f'File "{fname}" already exists! Skipping.', end="\r")
                continue
        else:
            im = Image.fromarray( (255*(im/2**16)).astype(np.uint8))
            im.save(fname)
    return 

# %%
scale = 1.6
dir_name = f'../datasets/{dataset_name}_spotscale{int(100*scale)}'
gen = image.generate_spot_crops(adata, return_obs=True, as_array='image', spot_scale=scale)

# write_img_from_gen(gen, dir_name)

# %%
fname = os.listdir(dir_name)[0]
image = Image.open(f'{dir_name}/{fname}')
image

# %%
from torchvision.transforms import ToTensor

image = ToTensor()(image)

# %%
image.size()

# %%
import numpy as np 

files = os.listdir(dir_name)
data_module = ImageDataModule(np.array(files), dataset_dir = dir_name, test_split_seed= 42, val_split_seed = 42,)

# %%
data_module

# %%

