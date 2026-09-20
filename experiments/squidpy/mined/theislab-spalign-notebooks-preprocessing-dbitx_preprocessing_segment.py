# mined from: https://github.com/theislab/spalign/blob/74113d4f8ef459f867ebadc1ca83f99dfaa3948c/notebooks/preprocessing/dbitx_preprocessing_segment.ipynb
# symbols: squidpy.im.ImageContainer, squidpy.im.segment

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

from cellpose import models

# %%
IMAGE_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'images'
SEG_IMAGE_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'seg_images' 
ADATA_DIR = PROJECT_DIR / 'datasets' / 'spatial' / 'adata'

# %%
list(IMAGE_DIR.iterdir())

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
channels = {experiment_id:['dapi', 'phalloidin', *chdicts[experiment_id.split('_')[2]].values()]for experiment_id in experiment_ids}

# %%
gpu = torch.cuda.is_available() 
print(f'gpu: {gpu}')

def cellpose(img, min_size=15):
    """
    Segment a nuclear image with CellPose 
    """
    model = models.Cellpose(model_type='nuclei', gpu=gpu)
    res, _, _, _ = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        min_size=min_size,
    )
    return res

# %%
for experiment_id in tqdm(os.listdir(IMAGE_DIR)):
    if experiment_id != 'high_res':
        # Read image container 
        container = sq.im.ImageContainer(IMAGE_DIR / experiment_id)
        # Read 
        sq.im.segment(img=container, layer="dapi",  method=cellpose)
        # Save image
        save_path = experiment_id+'_segmented'
        container.save(SEG_IMAGE_DIR / save_path)
        del container

# %%
container = sq.im.ImageContainer(SEG_IMAGE_DIR / '37_48_A1_segmented')

# %%
crop = container.crop_corner(15000, 15000, 10000)
crop.show(layer='segmented_custom')
