# mined from: https://github.com/theislab/stcti_notebook/blob/92f8c5497a6a6e3b6f3749c9905f5782b0244fed/image_preprocessing/pp_spot_masks.ipynb
# symbols: squidpy.im.ImageContainer

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt 
import scanpy as sc
import squidpy as sq


from tqdm.notebook import tqdm
from pathlib import Path
from scipy.ndimage.measurements import center_of_mass
from skimage.io import imread, imsave
from stcti.paths import DATA_DIR, PROJECT_DIR

# %%
data_type = 'human_brain_V1'
img_path = DATA_DIR / data_type / 'segmentation'
# img_path = PROJECT_DIR / 'brain_images'/ 'segmentation'

# %%
image = sq.im.ImageContainer().load(img_path)

# %%
image

# %%
n_channels = 1


crop = image.crop_corner(10300, 13300, size=300)

fig, ax = plt.subplots(1, n_channels, figsize=(5,5))
crop.show(channelwise=True, layer='segmented_custom', ax=ax)

# %%
def extract_seg_mask(img, val, max_dist=2000, img_size=64, save=False, folder=None, verbose=False):    
    idx = np.array(np.where(seg==val))
    
    y_min = idx[0].min()
    y_max = idx[0].max()
    dist0 = y_max - y_min

    x_min = idx[1].min()
    x_max = idx[1].max()
    dist1 = x_max - x_min

    coms = []
    masks = []
    
    # Check for multiple modes, that is multiple masks with the same value
    # I assume that there are no more than 2 masks with the same value
    mask = [np.zeros(idx.shape[1]) == 0]
    if max(dist0, dist1) > max_dist: 
        cond = idx[0] < (idx[0].max()-max_dist)
        mask = [cond, ~cond]
    
    for m in mask:
        shift = idx[:, m].min(1)
        _idx = idx[:, m] - shift.reshape([2,1])
        tmp = np.zeros((img_size,img_size))
        if _idx.max() >= img_size-1: 
            return (val)
        tmp[_idx[0], _idx[1]]= 255

        com_y, com_x = np.array(center_of_mass(tmp)).astype(int)

        _idx += img_size//2 
        _idx -= [[com_y], [com_x]]
        if _idx.max() >= img_size-1: 
            return val
        tmp = np.zeros((img_size,img_size))
        tmp[_idx[0], _idx[1]]= 255
        com = [com_y, com_x] + shift
        coms.append(tuple(com))
        # Save 
        if save:
            assert isinstance(folder, Path)
            f_name = 'com_' + str(com[0]) + '_' + str(com[1]) 
            imsave(folder/(f_name + '.tif'), tmp.astype('uint8'))
        else: 
            masks.append(tmp.copy())
    if verbose:
        print(f'Found {len(coms)} segmentation mask with value {val}.')
    if not save: 
        return dict(zip(coms, masks))

# %%
folder = PROJECT_DIR / 'brain_images'/ 'seg_masks'
if not folder.exists(): 
    print(f"Creating folder: {folder}")
    os.mkdir(folder)

# %%
# !rm -rf /mnt/home/icb/leon.hetzel/git/stcti/datasets/dbitx_kidney/seg_masks

# %%
seg = image['segmented_custom']
uniques = np.unique(seg)
uniques

# %%
len(uniques)

# %%
seg_masks = {}

for i in tqdm(range(len(uniques[:19]))): 
    val = uniques[i]
    if val == 0: 
        continue
    seg = seg.squeeze()
    seg_masks_tmp = extract_seg_mask(seg, val, max_dist=2000, img_size=64, verbose=True)
    seg_masks = {**seg_masks, **seg_masks_tmp}

# %%
rows = 3 
cols = 6
assert rows*cols <= len(seg_masks), print(f'Number of masks: {len(seg_masks)}')

fig, ax = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

for i, (com,m) in enumerate(seg_masks.items()):
    if i == cols*rows: 
        break
    ax[i//cols, i%cols].imshow(m)
    ax[i//cols, i%cols].set_title(f"com: {com}")

# %%
from joblib import Parallel, delayed

results = Parallel(n_jobs=48)(delayed(extract_seg_mask)(seg, val, max_dist=2000, img_size=64, folder=folder, save=True) for val in uniques[1:])

large_masks_idx = [x for x in results if x is not None]

# %%
print(f'Found {len(large_masks_idx)} cells that are too large for 64x64.')

# %%
big_masks = {}

for i in tqdm(range(len(large_masks_idx))): 
    val = large_masks_idx[i]
    if val == 0: 
        continue
    seg = seg.squeeze()
    big_masks_tmp = extract_seg_mask(seg, val, max_dist=2000, img_size=128)
    if not isinstance(big_masks_tmp, dict): 
        print(big_masks_tmp)
        continue
    big_masks = {**big_masks, **big_masks_tmp}

# %%
rows = 3 
cols = 6 
assert rows*cols <= len(seg_masks), print(f'Number of masks: {len(seg_masks)}')

fig, ax = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

for i, m in enumerate(big_masks.values()):
    if i == cols*rows: 
        break
    ax[i//cols, i%cols].imshow(m)

# %%
# largest_masks_idx = [29513, 29820, 29934, 30051, 30066, 30457, 45051, 45721, 46190, 59126, 136248]
largest_masks_idx = []

biggest_masks = {}

for i in tqdm(range(len(largest_masks_idx))): 
    val = largest_masks_idx[i]
    if val == 0: 
        continue
    seg = seg.squeeze()
    biggest_masks_tmp = extract_seg_mask(seg, val, max_dist=2000, img_size=256)
    if not isinstance(big_masks_tmp, dict): 
        print(big_masks_tmp)
        continue
    biggest_masks = {**biggest_masks, **biggest_masks_tmp}

# %%
rows = 2 
cols = 6 

fig, ax = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

for i, (com, m) in enumerate(biggest_masks.items()):
    ax[i//cols, i%cols].imshow(m)
    ax[i//cols, i%cols].set_title(f'COM: {com}')
    ax[i//cols, i%cols].axis('off')
for j in range(i+1, rows*cols):
    ax[j//cols, j%cols].set_visible(False)

# %%
big_masks = {**big_masks, **biggest_masks}

# %%
for com, mask in tqdm(big_masks.items()): 
    f_name = 'com_' + str(com[0]) + '_' + str(com[1]) 
    imsave(folder/(f_name + '.tif'), mask.astype('uint8'))

# %%
_img = imread(folder / (f_name + '.tif'))
plt.imshow(_img)

# %%
folder

# %%


# %%

