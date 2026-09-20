# mined from: https://github.com/theislab/spalign/blob/74113d4f8ef459f867ebadc1ca83f99dfaa3948c/notebooks/preprocessing/dbitx_preprocessing_cell_coordinates.ipynb
# symbols: squidpy.im.ImageContainer

# %%
from spalign.paths import PROJECT_DIR
from spalign.preprocessing.image_utils import get_scaler, crop_spot_images, tensor_to_image
import squidpy as sq
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms.functional as fn
from tqdm import tqdm 
import tifffile as tiffio
import os
import skimage
from tqdm import tqdm 
import pandas as pd

from cellpose import models

# %%
SEG_IMAGE_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'seg_images' 
ADATA_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'adata'
METADATA_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'metadata'

# %%
# Read the anndata 
adata = sc.read(ADATA_DIR / "37_48_adata_pp_wohires.h5ad")

# %%
cell_coordinate_df = pd.DataFrame({'slide':[],
                                   'label':[], 
                                   'spot':[], 
                                   'local_y':[], 
                                   'local_x':[], 
                                   'local_y':[], 
                                   'global_y': []})

for experiment_id in tqdm(os.listdir(SEG_IMAGE_DIR)):
    container = sq.im.ImageContainer(SEG_IMAGE_DIR / experiment_id)
    experiment_id = experiment_id.replace("_segmented", "")
    
    # Subset adata
    data_sub = adata[adata.obs['id'] == experiment_id].copy()
    # Get global coordinates from segmentation mask 
    global_coordinates = pd.DataFrame(skimage.measure.regionprops_table(np.array(container['segmented_custom'].squeeze()), 
                                     properties=('label',
                                                 'centroid'))).set_index('label')
    # Change names in the data frame to make them informative
    global_coordinates.columns = ['global_x', 'global_y']

    # Generate spot crops
    spots = container.generate_spot_crops(data_sub, 
                                        library_id=f'{experiment_id}-dapi', 
                                        spot_diameter_key='spot_diameter_real', 
                                        return_obs = True, 
                                        spot_scale=1)
    del container  # free memory

    for spot in spots:
        spot_coord = skimage.measure.regionprops_table(np.array(spot[0]['segmented_custom'].squeeze()), 
                                         properties=('label',
                                                     'centroid'))
        # Change the keys
        spot_coord['local_x'] = spot_coord.pop('centroid-0')
        spot_coord['local_y'] = spot_coord.pop('centroid-1')

        # Spot name as a column 
        spot_coord['spot'] = [spot[1] for _ in range(len(spot_coord['label']))]
        spot_coord['slide'] = [experiment_id for _ in range(len(spot_coord['label']))]

        # Cell id as extra column
        spot_coord['cell_id'] = [f'{spot_name}_{label}' for spot_name, label in zip(spot_coord['spot'], spot_coord['label'])]

        # Finally add global coordinates
        spot_coord['global_x'] = list(global_coordinates.loc[spot_coord['label']].global_x)
        spot_coord['global_y'] = list(global_coordinates.loc[spot_coord['label']].global_y)

        cell_coordinate_df = pd.concat([cell_coordinate_df, pd.DataFrame(spot_coord)], axis=0)

# %%
cell_coordinate_df.to_csv(METADATA_DIR / 'cell_coordinates.csv')

# %%
cell_coordinate_df

# %%
experiment_ids = np.unique(cell_coordinate_df.slide)
experiment_ids

# %%
for experiment_id in experiment_ids:
    data_sub = adata[adata.obs['id'] == experiment_id].copy()
    # Read one container per experiment 
    container = sq.im.ImageContainer(SEG_IMAGE_DIR / (experiment_id+'_segmented'))
        
    # Generate spots 
    spots = container.generate_spot_crops(data_sub, 
                                    library_id=f'{experiment_id}-dapi', 
                                    spot_diameter_key='spot_diameter_real', 
                                    return_obs = True, 
                                    spot_scale=1.5)
    
    for i, spot in enumerate(spots):
        c, a = spot
        subset_coord = cell_coordinate_df.loc[cell_coordinate_df.spot == a]
        for local_x, local_y in zip(subset_coord.local_x, subset_coord.local_y):
            crop = c.crop_center(int(local_x+199*0.25), int(local_y+199*0.25),  50)
            crop.show(layer='segmented_custom')
        break

# %%

