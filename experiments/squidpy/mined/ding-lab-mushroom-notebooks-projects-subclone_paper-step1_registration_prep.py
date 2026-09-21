# mined from: https://github.com/ding-lab/mushroom/blob/dc3598950fd233f2255a8e53006c915cc425a4a7/notebooks/projects/subclone_paper/step1_registration_prep.ipynb
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

# %%
run_dir = '/diskmnt/Projects/Users/estorrs/mushroom/data/projects/subclone'
Path(run_dir).mkdir(parents=True, exist_ok=True)

# %%
reg_dir = os.path.join(run_dir, 'registration')
Path(reg_dir).mkdir(parents=True, exist_ok=True)

# %%
data_map = {
    'HT397B1': {
        'order': [
            'HT397B1-U1',
            'HT397B1-U2',
            'HT397B1-U12',
            'HT397B1-U21',
            'HT397B1-U22',
            'HT397B1-U31'
        ],
        'data': {
            'visium': {
                'HT397B1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/HT397B1/S1H3/HT397B1-S1H3A1U1Bp1/outs',
                'HT397B1-U21': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/HT397B1/S1H3/HT397B1-S1H3A1U21Bp1/outs',
            },
            'he': {
                'HT397B1-U1':  '/diskmnt/Datasets/Spatial_Transcriptomics/images/all/A1_HT397B1-S1H3A1U1.tif',
                'HT397B1-U21': '/diskmnt/Datasets/Spatial_Transcriptomics/images/all/B1_HT397B1-S1H3A1U21.tif',
            },
            'multiplex': {
                'HT397B1-U2': '/diskmnt/primary/CODEX/HTAN/031623_BRCA_HT397B1-U2/HT397B1-S1H3A1-U2__20230315.ome.tiff',
                'HT397B1-U12': '/diskmnt/primary/CODEX/HTAN/03172023_BRCA_HT397B1-U12/HT397B1-S1H3A1-U22__20230316.ome.tiff',
                'HT397B1-U22': '/diskmnt/primary/CODEX/HTAN/041223_BRCA_HT397B1-S1H3A1-U22/HT397B1-S1H3A1-U22__20230413.ome.tiff',
                'HT397B1-U31': '/diskmnt/primary/CODEX/HTAN/040623_BRCA_HT397B1-U31/HT397B1-S1H3A1-U31__20230407.ome.tiff',
            }
        }
    },
    # look at HT271B1-S1H3Fc2U1Z1Bs1 for codex alignment
    #!!! 413 has codex
    'HT206B1': {
        'order': [
            'HT206B1-U2',
            'HT206B1-U3',
            'HT206B1-U4',
            'HT206B1-U5',
        ],
        'data': {
            'visium': {
                'HT206B1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT206B1/H1/HT206B1-S1Fc1U2Z1B1/outs',
                'HT206B1-U3': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT206B1/H1/HT206B1-S1Fc1U3Z1B1/outs',
                'HT206B1-U4': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT206B1/H1/HT206B1-S1Fc1U4Z1B1/outs',
                'HT206B1-U5': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT206B1/H1/HT206B1-S1Fc1U5Z1B1/outs',
            },
        }
    },
    'HT268B1': {
        'order': [
            'HT268B1-U1',
            'HT268B1-U2',
            'HT268B1-U12',
            'HT268B1-U22',
            'HT268B1-U32',
        ],
        'data': {
            'visium': {
                'HT268B1-U12': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT268B1/Th1H3/HT268B1-Th1H3Fc2U12Z1Bs1/outs',
                'HT268B1-U22': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT268B1/Th1H3/HT268B1-Th1H3Fc2U22Z1Bs1/outs',
                'HT268B1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT268B1/Th1H3/HT268B1-Th1H3Fc2U2Z1Bs1/outs',
                'HT268B1-U32': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT268B1/Th1H3/HT268B1-Th1H3Fc2U32Z1Bs1/outs',
                'HT268B1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT268B1/H1/HT268B1-Th1K3Fc2U1Z1Bs1/outs',
            },
        }
    },
    'HT225C1': {
        'order': [
            'HT225C1-U1',
            'HT225C1-U2',
            'HT225C1-U3',
            'HT225C1-U4',
            'HT225C1-U5',
        ],
        'data': {
            'visium': {
                'HT225C1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U1Z1B1/outs',
                'HT225C1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U2Z1B1/outs',
                'HT225C1-U3': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U3Z1B1/outs',
                'HT225C1-U4': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U4Z1B1/outs',
                'HT225C1-U5': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U5Z1B1/outs',
            },
        }
    },
    'HT243B1-H3': {
        'order': [
            'HT243B1-H3-U1',
            'HT243B1-H3-U2',
            'HT243B1-H3-U3',
            'HT243B1-H3-U4',
        ],
        'data': {
            'visium': {
                'HT243B1-H3-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT243B1/H3/HT243B1H3A2-S1Fc1U1Z1B1/outs',
                'HT243B1-H3-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT243B1/H3/HT243B1H3A2-S1Fc1U2Z1B1/outs',
                'HT243B1-H3-U3': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT243B1/H3/HT243B1H3A2-S1Fc1U3Z1B1/outs',
                'HT243B1-H3-U4': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT243B1/H3/HT243B1H3A2-S1Fc1U4Z1B1/outs'
            },
        }
    },
    'HT271B1': {
        'order': [
            'HT271B1-U1',
            'HT271B1-U2',
            'HT271B1-U3',
            'HT271B1-U4',
        ],
        'data': {
            'visium': {
                'HT271B1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT271B1/H3/HT271B1-S1H3Fc2U1Z1Bs1/outs',
                'HT271B1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT271B1/H3/HT271B1-S1H3Fc2U2Z1Bs1/outs',
                'HT271B1-U3': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT271B1/H3/HT271B1-S1H3Fc2U3Z1Bs1/outs',
                'HT271B1-U4': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT271B1/H3/HT271B1-S1H3Fc2U4Z1Bs1/outs',
            },
        }
    },
    'HT413C1': {
        'order': [
            'HT413C1-U1',
            'HT413C1-U2',
            'HT413C1-U14',
        ],
        'data': {
            'visium': {
                'HT413C1-U14': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/HT413C1/Th1K2/HT413C1-Th1K2A4U14Bp1/outs',
                'HT413C1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/HT413C1/Th1K2/HT413C1-Th1K2A4U2Bp1/outs',
                'HT413C1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/HT413C1/Th1K2/HT413C1-Th1K2A4U1Bp1/outs',
            },
        }
    },
    'HT112C1': {
        'order': [
            'HT112C1-U1',
            'HT112C1-U2',
        ],
        'data': {
            'visium': {
                'HT112C1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT112C1/H1/HT112C1-U1_ST_Bn1/outs',
                'HT112C1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT112C1/H1/HT112C1-U2_ST_Bn1/outs',
            },
        }
    },
    'HT226C1': {
        'order': [
            'HT226C1-U1',
            'HT226C1-U2',
        ],
        'data': {
            'visium': {
                'HT226C1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT226C1/H1/HT226C1-Th1Fc1U1Z1Bn1/outs',
                'HT226C1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT226C1/H1/HT226C1-Th1Fc1U2Z1Bn1/outs',
            },
        }
    },
    'HT235B1': {
        'order': [
            'HT235B1-U1',
            'HT235B1-U2',
        ],
        'data': {
            'visium': {
                'HT235B1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT235B1/H1/HT235B1-S1Fc1U1Z1Bn1/outs',
                'HT235B1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT235B1/H1/HT235B1-S1Fc1U2Z1Bn1/outs',
            },
        }
    },
    'HT243B1-H4': {
        'order': [
            'HT243B1-H4-U1',
            'HT243B1-H4-U2',
        ],
        'data': {
            'visium': {
                'HT243B1-H4-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT243B1/H4/HT243B1H4A2-S1Fc1U1Z1B1/outs',
                'HT243B1-H4-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT243B1/H4/HT243B1H4A2-S1Fc1U2Z1B1/outs',
            },
        }
    },
    'HT339B1': {
        'order': [
            'HT339B1-U1',
            'HT339B1-U2',
        ],
        'data': {
            'visium': {
                'HT339B1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT339B1/H3/HT339B1-S1H3Fc2U1Z1Bs1/outs',
                'HT339B1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT339B1/H3/HT339B1-S1H3Fc2U2Bs2/outs',
            },
        }
    },
    'HT448C1': {
        'order': [
            'HT448C1-U1',
            'HT448C1-U13',
        ],
        'data': {
            'visium': {
                'HT448C1-U13': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/HT448C1/Th1K1/HT448C1-Th1K1Fp1U13Bp1/outs',
                'HT448C1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/HT448C1/Th1K1/HT448C1-Th1K1Fp1U1Bp1/outs',
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
scale = .1

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
                
            if dtype == 'he':
                x = tifffile.imread(filepath)
                x = rescale(x, scale=scale)

                tifffile.imwrite(os.path.join(output_dir, f's{idx}.tif'), x, compression='LZW')

# %%
metadata = {}
for case, d in data_map.items(): 
    output_dir = os.path.join(reg_dir, case)
    
    adata = sq.read.visium(next(iter(d['data']['visium'].values())))
    scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
    registered_pixels_per_micron = scalefactors['spot_diameter_fullres'] / 65. # each spot is 65 microns wide
    
    d.update({
        'ids': [f's{i}' for i in range(len(d['order']))],
        'scale': scale,
        'registered_pixels_per_micron': registered_pixels_per_micron
    })
    metadata[case] = d

# %%
yaml.safe_dump(metadata, open(os.path.join(reg_dir, 'metadata.yaml'), 'w'))

# %%

