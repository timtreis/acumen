# mined from: https://github.com/ding-lab/mushroom/blob/dc3598950fd233f2255a8e53006c915cc425a4a7/notebooks/projects/xenium_grant/step1_registration_prep.ipynb
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
from mushroom.data.multiplex import extract_ome_tiff, get_ome_tiff_channels, make_pseudo
from mushroom.data.xenium import adata_from_xenium

# %%
run_dir = '/diskmnt/Projects/Users/estorrs/mushroom/data/projects/xenium_grant_v2'
Path(run_dir).mkdir(parents=True, exist_ok=True)

# %%
reg_dir = os.path.join(run_dir, 'registration')
Path(reg_dir).mkdir(parents=True, exist_ok=True)

# %%
data_map = {
    'HT206B1': {
        'order': [
            'HT206B1-U1',
            'HT206B1-U2',
            'HT206B1-U5',
            'HT206B1-U8',
            'HT206B1-U9',
            'HT206B1-U10',
            'HT206B1-U13',
            'HT206B1-U16',
            'HT206B1-U17',
            'HT206B1-U18',
            'HT206B1-U21',
            'HT206B1-U24',
        ],
        'data': {
            'xenium': {
                'HT206B1-U24': '/diskmnt/primary/Xenium/data/20230830__153957__20230830_24001/output-XETG00122__0010528__HT206B1-H2L1Us1_1__20230830__154053',
                'HT206B1-U16': '/diskmnt/primary/Xenium/data/20230830__153957__20230830_24001/output-XETG00122__0010528__HT206B1-H2L1Us1_9__20230830__154053',
                'HT206B1-U8': '/diskmnt/primary/Xenium/data/20230830__153957__20230830_24001/output-XETG00122__0010528__HT206B1-H2L1Us1_17__20230830__154053',
                'HT206B1-U1': '/diskmnt/primary/Xenium/data/20230919__220553__24003/output-XETG00122__0010520__HT206B1-H2L1Us1_8__20230919__220650',
                'HT206B1-U9': '/diskmnt/primary/Xenium/data/20230919__220553__24003/output-XETG00122__0010520__HT206B1-H2L1Us1_15__20230919__220650',
                'HT206B1-U17': '/diskmnt/primary/Xenium/data/20230919__220553__24003/output-XETG00122__0010520__HT206B1-H2L1Us1_24__20230919__220650',
                
            },
            'multiplex': {
                'HT206B1-U2': '/diskmnt/primary/CODEX/HTAN/20230914_BRCA_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U2__20230914.ome.tiff',
                'HT206B1-U10': '/diskmnt/primary/CODEX/HTAN/20230914_BRCA_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U10__20230914.ome.tiff',
                'HT206B1-U18': '/diskmnt/primary/CODEX/HTAN/20230914_BRCA_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U18__20230914.ome.tiff',
                'HT206B1-U5': '/diskmnt/primary/CODEX/HTAN/20231002_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U5__20231002.ome.tiff',
                'HT206B1-U13': '/diskmnt/primary/CODEX/HTAN/20231002_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U13__20231002.ome.tiff',
                'HT206B1-U21': '/diskmnt/primary/CODEX/HTAN/20231002_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U21__20231002.ome.tiff',
            }
        }
    },
    'S18-9906': {
        'order': [
            'S18-9906-U2',
            'S18-9906-U3',
            'S18-9906-U9',
            'S18-9906-U10',
            'S18-9906-U17',
            'S18-9906-U18',
            'S18-9906-U24',
            'S18-9906-U25',
        ],
        'data': {
            'xenium': {
                'S18-9906-U2': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010784__S18-9906-B27Us1_2Q1__20230912__220421',
                'S18-9906-U17': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010784__S18-9906-B27Us1_17Q1__20230912__220421',
                'S18-9906-U9': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010787__S18-9906-B27Us1_9Q1__20230912__220421',
                'S18-9906-U24': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010787__S18-9906-B27Us1_24Q1__20230912__220421',
                
            },
            'multiplex': {
                'S18-9906-U3': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U3__20231006.ome.tiff',
                'S18-9906-U10': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U10__20231006.ome.tiff',
                'S18-9906-U18': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U18__20231006.ome.tiff',
                'S18-9906-U25': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U25__20231006.ome.tiff',
            }
        }
    },
    'S18-25943': {
        'order': [
            'S18-25943-U1',
#             'S18-25943-U2',
            'S18-25943-U4',
            'S18-25943-U8',
            'S18-25943-U9',
            'S18-25943-U11',
            'S18-25943-U13',
#             'S18-25943-U16',
        ],
        'data': {
            'xenium': {
                'S18-25943-U1': '/diskmnt/primary/Xenium/data/20231117__205826__24011/output-XETG00122__0011123__S18-25943-A7Us1_1__20231117__205842',
                'S18-25943-U8': '/diskmnt/primary/Xenium/data/20231117__205826__24011/output-XETG00122__0011123__S18-25943-A7Us1_8__20231117__205843',
                'S18-25943-U4': '/diskmnt/primary/Xenium/data/20231117__205826__24011/output-XETG00122__0011128__S18-25943-A7Us1_4__20231117__205843',
                'S18-25943-U11': '/diskmnt/primary/Xenium/data/20231117__205826__24011/output-XETG00122__0011128__S18-25943-A7Us1_11__20231117__205843',
            },
            'multiplex': { # only taking the middle because top and bottom piece are cut off
#                 'S18-25943-U2': '/diskmnt/primary/CODEX/HTAN/20231122_Human_pancreatic_cancer_S18-25943-A7Us1_2__Us1_9__Us1_13__Us1_16/S18-25943-U2__20231122.ome.tiff',
                'S18-25943-U9': '/diskmnt/primary/CODEX/HTAN/20231122_Human_pancreatic_cancer_S18-25943-A7Us1_2__Us1_9__Us1_13__Us1_16/S18-25943-U9__20231122.ome.tiff',
                'S18-25943-U13': '/diskmnt/primary/CODEX/HTAN/20231122_Human_pancreatic_cancer_S18-25943-A7Us1_2__Us1_9__Us1_13__Us1_16/S18-25943-U13__20231122.ome.tiff',
#                 'S18-25943-U16': '/diskmnt/primary/CODEX/HTAN/20231122_Human_pancreatic_cancer_S18-25943-A7Us1_2__Us1_9__Us1_13__Us1_16/S18-25943-U16__20231122.ome.tiff',
            }
        }
    },
    'S18-5591': {
        'order': [
            'S18-5591-U1',
#             'S18-5591-U2',
            'S18-5591-U5',
#             'S18-5591-U6',
            'S18-5591-U7',
            'S18-5591-U8',
            'S18-5591-U12',
            'S18-5591-U14',
            'S18-5591-U18',
            'S18-5591-U19',
            'S18-5591-U20',
            'S18-5591-U21',
            'S18-5591-U23',
            'S18-5591-U24',
        ],
        'data': {
            'xenium': {
                'S18-5591-U1': '/diskmnt/primary/Xenium/data/20231114__223057__24010/output-XETG00122__0011055__S18-5591-C8Us2_1__20231114__223131',
                'S18-5591-U7': '/diskmnt/primary/Xenium/data/20231114__223057__24010/output-XETG00122__0011055__S18-5591-C8Us2_7__20231114__223131',
                'S18-5591-U18': '/diskmnt/primary/Xenium/data/20231114__223057__24010/output-XETG00122__0011055__S18-5591-C8Us2_18__20231114__223131',
                'S18-5591-U5': '/diskmnt/primary/Xenium/data/20231114__223057__24010/output-XETG00122__0010977__S18-5591-C8Us2_5__20231114__223131',
                'S18-5591-U12': '/diskmnt/primary/Xenium/data/20231114__223057__24010/output-XETG00122__0010977__S18-5591-C8Us2_12__20231114__223131',
                'S18-5591-U20': '/diskmnt/primary/Xenium/data/20231114__223057__24010/output-XETG00122__0010977__S18-5591-C8Us2_20__20231114__223131',
            },
            'multiplex': { # top pieces are cut out for both runs
#                 'S18-5591-U2': '/diskmnt/primary/CODEX/HTAN/20231116_Human_prostate_African_American_serial_S18_5591_Slide_2/S18-5591-U2__20231116.ome.tiff',
                'S18-5591-U8': '/diskmnt/primary/CODEX/HTAN/20231116_Human_prostate_African_American_serial_S18_5591_Slide_2/S18-5591-U8__20231116.ome.tiff',
                'S18-5591-U19': '/diskmnt/primary/CODEX/HTAN/20231116_Human_prostate_African_American_serial_S18_5591_Slide_2/S18-5591-U19__20231116.ome.tiff',
                'S18-5591-U23': '/diskmnt/primary/CODEX/HTAN/20231116_Human_prostate_African_American_serial_S18_5591_Slide_2/S18-5591-U23__20231116.ome.tiff',
#                 'S18-5591-U6': '/diskmnt/primary/CODEX/HTAN/20231118_Human_prostate_African_American_serial_S18_5591_Slide_6/S18-5591-U6__20231118.ome.tiff',
                'S18-5591-U14': '/diskmnt/primary/CODEX/HTAN/20231118_Human_prostate_African_American_serial_S18_5591_Slide_6/S18-5591-U14__20231118.ome.tiff',
                'S18-5591-U21': '/diskmnt/primary/CODEX/HTAN/20231118_Human_prostate_African_American_serial_S18_5591_Slide_6/S18-5591-U21__20231118.ome.tiff',
                'S18-5591-U24': '/diskmnt/primary/CODEX/HTAN/20231118_Human_prostate_African_American_serial_S18_5591_Slide_6/S18-5591-U24__20231118.ome.tiff',
            }
        }
    }
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
    'Pan-Cytokeratin': ['Pan-Cytokeratin', 'Pan-CK', 'Pan-CK (D)', 'PanCK (D)'],
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
    
    adata = adata_from_xenium(next(iter(d['data']['xenium'].values())))
    scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
    registered_pixels_per_micron = scalefactors['tissue_hires_scalef'] # when read in coords are in microns, so hires_scalef is ppm
    
    d.update({
        'ids': [f's{i}' for i in range(len(d['order']))],
        'scale': scale,
        'registered_pixels_per_micron': registered_pixels_per_micron
    })
    metadata[case] = d

# %%
yaml.safe_dump(metadata, open(os.path.join(reg_dir, 'metadata.yaml'), 'w'))

# %%
metadata

# %%

