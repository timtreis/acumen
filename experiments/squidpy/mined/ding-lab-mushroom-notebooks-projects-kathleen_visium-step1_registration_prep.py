# mined from: https://github.com/ding-lab/mushroom/blob/dc3598950fd233f2255a8e53006c915cc425a4a7/notebooks/projects/kathleen_visium/step1_registration_prep.ipynb
# symbols: squidpy.read.visium

# %%
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import tifffile
import torch
import torchvision.transforms.functional as TF
import yaml
from einops import rearrange

# %%
# %load_ext autoreload

# %%
# %autoreload 2

# %%
from mushroom.data.multiplex import extract_ome_tiff, get_ome_tiff_channels, make_pseudo, pixels_per_micron
import mushroom.data.multiplex as multiplex
import mushroom.data.visium as visium
from mushroom.data.xenium import adata_from_xenium

# %%
run_dir = '/diskmnt/Projects/Users/estorrs/mushroom/data/projects/kathleen_visium'
Path(run_dir).mkdir(parents=True, exist_ok=True)

# %%
reg_dir = os.path.join(run_dir, 'registration')
Path(reg_dir).mkdir(parents=True, exist_ok=True)

# %%
data_map = {
    '17B41236A': {
        'order': [
            '17B41236A-A',
            '17B41236A-B',
            '17B41236A-C',
            '17B41236A-D',
        ],
        'data': {
            'visium': {
                '17B41236A-A': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/kathleen/st/PC_A_spatial_outs/outs',
                '17B41236A-B': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/kathleen/st/PC_B_spatial_outs/outs',
                '17B41236A-C': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/kathleen/st/PC_C_spatial_outs/outs',
                '17B41236A-D': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/kathleen/st/PC_D_spatial_outs/outs',
            },
        }
    },
}

# %%
def rescale(x, scale=.1):
    x = rearrange(torch.tensor(x), 'h w c -> c h w')
    x = TF.resize(x, (int(x.shape[-2] * scale), int(x.shape[-1] * scale)), antialias=True)
    x = TF.convert_image_dtype(x, torch.uint8)
    x = rearrange(x.numpy(), 'c h w -> h w c')
    
    return x

# %%
scale = .2

# %%
official_to_options = {
    'Pan-Cytokeratin': ['Pan-Cytokeratin', 'Pan-CK', 'Pan-CK (D)', 'PanCK (D)', 'PanCytokeratin'],
    'CD45': ['CD45 (D)', 'CD45', 'CD45-(D)'],
    'DAPI': ['DAPI'],
    'SMA': ['SMA-(D)', 'SMA', 'SMA (D)', 'a-SMA (D)'],
}
channel_mapping = {v:k for k, vs in official_to_options.items() for v in vs}

cmap = {
    'DAPI': (0., 0., 1.),
    'Pan-Cytokeratin': (1., 0., 0.),
    'CD45': (0., 1., 0.),
    'SMA': (1., 1., 1.)
}

# %%
for case, d in data_map.items():
    output_dir = os.path.join(reg_dir, case, 'unregistered_tifs')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    order = d['order']
    
    for dtype, data in d['data'].items():
        for sid, filepath in data.items():
            print(dtype, sid)
            idx = order.index(sid)
            if dtype == 'visium':
                adata = sq.read.visium(filepath)

                d = next(iter(adata.uns['spatial'].values()))
                he = d['images']['hires']
                sf = scale / d['scalefactors']['tissue_hires_scalef']
                he = rescale(he, scale=sf)
                
                tifffile.imwrite(os.path.join(output_dir, f's{idx}.tif'), he, compression='LZW')
                
            if dtype == 'multiplex':
                channels = get_ome_tiff_channels(filepath)
                keep = [c for c in channels if channel_mapping.get(c, c) in cmap]
                d = extract_ome_tiff(filepath, channels=keep)
                d = {channel_mapping[channel]:np.squeeze(rescale(np.expand_dims(img, -1), scale=scale))
                     for channel, img in d.items()}

                pseudo = make_pseudo(d, cmap=cmap, contrast_pct=90.)
                pseudo /= pseudo.max()
                pseudo *= 255
                pseudo = pseudo.astype(np.uint8)

                tifffile.imwrite(os.path.join(output_dir, f's{idx}.tif'), pseudo, compression='LZW')
                
            if dtype == 'xenium':
                adata = adata_from_xenium(filepath)
                
                d = next(iter(adata.uns['spatial'].values()))
                x = d['images']['hires']
                sf = scale / d['scalefactors']['tissue_hires_scalef']
                x = rescale(rearrange(x, 'h w -> h w 1'), scale=sf)
                x = x.astype(np.float32) / x.max()
                x *= 255.
                x = x.astype(np.uint8)
                
                tifffile.imwrite(os.path.join(output_dir, f's{idx}.tif'), x, compression='LZW')
                
            if dtype == 'he':
                x = tifffile.imread(filepath)
                x = rescale(x, scale=scale)

                tifffile.imwrite(os.path.join(output_dir, f's{idx}.tif'), x, compression='LZW')

# %%
metadata = {}
for case, d in data_map.items(): 
    output_dir = os.path.join(reg_dir, case)
    
    sample = d['order'][0]
    for k, v in d['data'].items():
        if sample in v:
            if k == 'xenium':
                adata = adata_from_xenium(v[sample])
                scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
                registered_pixels_per_micron = scalefactors['tissue_hires_scalef'] # when read in coords are in microns, so hires_scalef is ppm
            elif k == 'multiplex':
                registered_pixels_per_micron = multiplex.pixels_per_micron(v[sample])
            elif k == 'visium':
                registered_pixels_per_micron = visium.pixels_per_micron(v[sample])
            elif k == 'he':
                # for now will just register to the first xenium image
#                 registered_pixels_per_micron = multiplex.pixels_per_micron(next(iter(d['data']['multiplex'].values())))
                adata = adata_from_xenium(next(iter(d['data']['xenium'].values())))
                scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
                registered_pixels_per_micron = scalefactors['tissue_hires_scalef'] # when read in coords are in microns, so hires_scalef is ppm
            else:
                raise RuntimeError('he not implemented yet')
                

    d.update({
        'ids': [f's{i}' for i in range(len(d['order']))],
        'scale': scale,
        'registered_pixels_per_micron': registered_pixels_per_micron
    })
    metadata[case] = d

# %%
yaml.safe_dump(metadata, open(os.path.join(reg_dir, 'metadata.yaml'), 'w'))

# %%
os.path.join(reg_dir, 'metadata.yaml')

# %%
import sys
yaml.safe_dump(metadata, sys.stdout)

# %%
a = visium.adata_from_visium('/diskmnt/Projects/Users/estorrs/imaging-analysis/data/kathleen/st/PC_D_spatial_outs/outs')
a

# %%
visium.pixels_per_micron(a)

# %%

