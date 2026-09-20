# mined from: https://github.com/theislab/spalign/blob/74113d4f8ef459f867ebadc1ca83f99dfaa3948c/notebooks/preprocessing/dbitx_preprocessing_rgb.ipynb
# symbols: squidpy.im.ImageContainer

# %%
from spalign.paths import PROJECT_DIR
import os
from spalign.preprocessing.image_utils import get_scaler, crop_spot_images, prep_composite, tensor_to_image
import squidpy as sq
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms.functional as fn
from tqdm import tqdm 

# %%
IMAGE_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'images'
ADATA_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'adata'
SPOT_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'spot_images'
OBS_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'spot_keys'

# %%
chdicts = {
    "A1": {
        "555": "CD31",
        "488": "Nephrin"
    },
    "A2": {
        "555": "AQP1",
        "488": "CD45"
    },
    "A3": {
        "555": "AQP1",
        "488": "Nephrin"
    },
    "B1": {
        "555": "CD31",
        "488": "CD45"
    },
    "B2": {
        "555": "CD31",
        "488": "Nephrin"
    },
    "B3": {
        "555": "AQP1",
        "488": "CD45"
    },
    "C1": {
        "555": "AQP1",
        "488": "Nephrin"
    },
    "C2": {
        "555": "CD31",
        "488": "CD45"
    },
    "C3": {
        "555": "AQP1",
        "488": "CD45"
    }
}

# %%
experiment_wells = list(chdicts.keys())
experiment_ids = ["37_48_" + experiment_well for experiment_well in experiment_wells]
experiment_ids

# %%
channels = {experiment_id:['dapi', 'phalloidin', *chdicts[experiment_id.split('_')[2]].values()]for experiment_id in experiment_ids}

# %%
adata = sc.read(ADATA_DIR / "37_48_adata_pp_wohires.h5ad")
adata

# %%
from matplotlib.colors import LinearSegmentedColormap

cmaps = {
    # dapi 
    "blue": LinearSegmentedColormap("rgb", 
                                    {'blue':  [(0.0,  0.0, 0.0),
                                               (1.0,  1.0, 1.0)],         
                                     'red':   [(0.0,  0.0, 0.0),
                                               (1.0,  0.0, 0.0)],
                                     'green': [(0.0,  0.0, 0.0),
                                               (1.0,  0.0, 0.0)],
                                    }),
    # phalloidin
    "red": LinearSegmentedColormap("rgb", 
                                  {'blue':  [(0.0,  0.0, 0.0),
                                          (1.0,  0.0, 0.0)],
                                 'red':   [(0.0,  0.0, 0.0),
                                           (1.0,  1.0, 1.0)],
                                 'green': [(0.0,  0.0, 0.0),
                                           (1.0,  0.0, 0.0)],
        }),
    # cd45
    "green": LinearSegmentedColormap("rgb", 
                                     {'blue':  [(0.0,  0.0, 0.0),
                                               (1.0,  0.0, 0.0)],         
                                     'red':   [(0.0,  0.0, 0.0),
                                               (1.0,  0.0, 0.0)],
                                     'green': [(0.0,  0.0, 0.0),
                                               (1.0,  1.0, 1.0)],
        }),
    # cd31
    "cyan": LinearSegmentedColormap("rgb", 
                                    {'blue':  [(0.0, 0.0, 0.0),
                                               (1.0,  1.0, 1.0)],         
                                     'red':   [(0.0,  0.0, 0.0),
                                               (1.0,  0.0, 0.0)],
                                     'green': [(0.0,  0.0, 0.0),
                                               (1.0,  1.0, 1.0)],
                                    }),
    # aqp1
    "yellow": LinearSegmentedColormap("rgb",
                                        {'blue':  [(0.0,  0.0, 0.0),
                                                   (1.0,  0.0, 0.0)],         
                                         'red':   [(0.0,  0.0, 0.0),
                                                   (1.0,  1.0, 1.0)],
                                         'green': [(0.0,  0.0, 0.0),
                                                   (1.0,  1.0, 1.0)],
                                        }),
    # nephrin
    "magenta": LinearSegmentedColormap("rgb",
                                        {'blue':  [(0.0,  0.0, 0.0),
                                                   (1.0,  1.0, 1.0)],         
                                         'red':   [(0.0,  0.0, 0.0),
                                                   (1.0,  1.0, 1.0)],
                                         'green': [(0.0,  0.0, 0.0),
                                                   (1.0,  0.0, 0.0)],
                                        }),
}

# Get different RGB encodings 
get_cmap = {
    'dapi': cmaps['blue'],
    'phalloidin': cmaps['red'],
    'CD45': cmaps['green'],
    'CD31': cmaps['cyan'],
    'AQP1': cmaps['yellow'],
    'Nephrin': cmaps['magenta']
}

# %%
for experiment_id in tqdm(experiment_ids):
    print(f'Load experiment {experiment_id}')
    container = sq.im.ImageContainer(IMAGE_DIR / experiment_id)
    # Scale images 
    [container.apply(get_scaler(ch), layer = ch, copy = False) for ch in channels[experiment_id]]
    # Subset adata 
    adata_sub = adata[adata.obs['id'] == experiment_id].copy()
    # Crop spots 
    print(f'Fragmenting into spots')
    images, obs = crop_spot_images(adata_sub, 
                     container, 
                     channels[experiment_id], 
                     experiment_id, 
                     spot_diameter='spot_diameter_real', 
                     return_obs=True)
    print(images)
    
    del container
    # Collapse the grayscale channel into single RGB images 
    collapsed_images = []
        
    print(f'Converting to RGB images from {experiment_id}')
    for region, o in enumerate(obs):
        current_row = []
        for channel in images.keys():
            current_row.append(get_cmap[channel](images[channel][region]))
        collapsed_images.append(prep_composite(current_row))
    
    del images
    # Stack and save
    print(f'Stack and save images and observation names')
    spot_imgs = torch.stack(collapsed_images)
    del collapsed_images
    image_path = SPOT_DIR / f'{experiment_id}_spots.pt'
    obs_path = OBS_DIR / f'{experiment_id}_obs.npy'
    torch.save(spot_imgs, image_path)
    np.save(obs_path, np.array(obs))

# %%
adata.obs

# %%

