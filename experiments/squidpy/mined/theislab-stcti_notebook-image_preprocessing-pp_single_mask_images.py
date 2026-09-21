# mined from: https://github.com/theislab/stcti_notebook/blob/92f8c5497a6a6e3b6f3749c9905f5782b0244fed/image_preprocessing/pp_single_mask_images.ipynb
# symbols: squidpy.im.ImageContainer

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scanpy as sc
import squidpy as sq


from tqdm.notebook import tqdm
from pathlib import Path
from scipy.ndimage.measurements import center_of_mass
from skimage.io import imread, imsave
from stcti.paths import DATA_DIR, PROJECT_DIR

# %%
data_type = 'human_brain_V1'
source_path = DATA_DIR / data_type / 'features'
segments_path = DATA_DIR / data_type / 'seg_masks'
combined_img_path = PROJECT_DIR / data_type / 'seg_images'

# %%
f_name = 'segmentation_features.parquet'

df = pd.read_parquet(source_path/f_name)
df.head()

# %%
df['com_y'] = df['filename_msk'].apply(lambda s: int(s.split('_')[1]))
df['com_x'] = df['filename_msk'].apply(lambda s: int(s.split('_')[2].split('.')[0]))

# %%
df.head()

# %%
image = sq.im.ImageContainer().load(DATA_DIR / data_type / 'segmentation')

# %%
image

# %%
def save_individual_segments(filename_msk=None, com=None, folder=None, source_path=None, img_size=64, n_channel=4, save=True): 
    tmp = np.zeros((img_size, img_size, n_channel))
    
    crop = image.crop_center(*com, radius=32)
    crop_array = crop['image'].values.copy()
    crop_array = crop_array.squeeze()
#     crop_array = np.transpose(crop_array, (2,0,1))
    
    f_name = filename_msk
    seg_mask = imread(source_path / filename_msk)
    if seg_mask.shape[0] > 64: 
        return com
    tmp[..., :n_channel-1] = crop_array[:64, :64, :]
    tmp[..., n_channel-1] = seg_mask
#     tmp = np.transpose(tmp, [1,2,0])
    if save: 
        f_name = 'com_' + str(com[0]) + '_' + str(com[1]) + '_allchannel' 
        np.save(folder / f_name, tmp.astype('uint16'))

# %%
folder = combined_img_path 
if not folder.exists(): 
    folder.mkdir(parents=True)
    print(f"Creating folder: {folder}")

# %%
import h5py

# %%
for i, row in df.iloc[:10].iterrows():
    f_name = row['filename_msk']
    com = row[['com_y', 'com_x']]
    save_individual_segments(filename_msk=f_name, com=com, img_size=64, n_channel=5, folder=folder, source_path=segments_path, save=True) 

# %%
from joblib import Parallel, delayed

results = Parallel(n_jobs=48)(delayed(save_individual_segments)(filename_msk=row['filename_msk'], com=row[['com_y', 'com_x']], img_size=64, n_channel=5, folder=folder, source_path=segments_path, save=True) for _, row in df.iterrows())

large_masks_idx = [x for x in results if x is not None]

# %%
img = np.load(folder/ 'com_9167_8090_allchannel.npy')
fig, ax = plt.subplots(1, 4, figsize=(4*5,5))
ax[0].imshow(img[...,0])
ax[1].imshow(img[...,1])
ax[2].imshow(img[...,2])
ax[3].imshow(img[...,3])

# %%
# Check that image is uint16 encoded
img

# %%
len(os.listdir(folder))

# %%
folder

# %%

