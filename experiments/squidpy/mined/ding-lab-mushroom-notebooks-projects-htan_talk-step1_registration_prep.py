# mined from: https://github.com/ding-lab/mushroom/blob/dc3598950fd233f2255a8e53006c915cc425a4a7/notebooks/projects/htan_talk/step1_registration_prep.ipynb
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
import mushroom.data.xenium as xenium
import mushroom.data.cosmx as cosmx

# %%
run_dir = '/diskmnt/Projects/Users/estorrs/mushroom/data/projects/htan_talk'
Path(run_dir).mkdir(parents=True, exist_ok=True)

# %%
reg_dir = os.path.join(run_dir, 'registration')
Path(reg_dir).mkdir(parents=True, exist_ok=True)

# %%
data_map = {
#     'HT206B1': {
#         'order': [
#             'HT206B1-U1',
#             'HT206B1-U2',
#             'HT206B1-U5',
#             'HT206B1-U8',
#             'HT206B1-U9',
#             'HT206B1-U10',
#             'HT206B1-U13',
#             'HT206B1-U16',
#             'HT206B1-U17',
#             'HT206B1-U18',
#             'HT206B1-U21',
#             'HT206B1-U24',
#         ],
#         'data': {
#             'xenium': {
#                 'HT206B1-U24': '/diskmnt/primary/Xenium/data/20230830__153957__20230830_24001/output-XETG00122__0010528__HT206B1-H2L1Us1_1__20230830__154053',
#                 'HT206B1-U16': '/diskmnt/primary/Xenium/data/20230830__153957__20230830_24001/output-XETG00122__0010528__HT206B1-H2L1Us1_9__20230830__154053',
#                 'HT206B1-U8': '/diskmnt/primary/Xenium/data/20230830__153957__20230830_24001/output-XETG00122__0010528__HT206B1-H2L1Us1_17__20230830__154053',
#                 'HT206B1-U1': '/diskmnt/primary/Xenium/data/20230919__220553__24003/output-XETG00122__0010520__HT206B1-H2L1Us1_8__20230919__220650',
#                 'HT206B1-U9': '/diskmnt/primary/Xenium/data/20230919__220553__24003/output-XETG00122__0010520__HT206B1-H2L1Us1_15__20230919__220650',
#                 'HT206B1-U17': '/diskmnt/primary/Xenium/data/20230919__220553__24003/output-XETG00122__0010520__HT206B1-H2L1Us1_24__20230919__220650',
                
#             },
#             'multiplex': {
#                 'HT206B1-U2': '/diskmnt/primary/CODEX/HTAN/20230914_BRCA_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U2__20230914.ome.tiff',
#                 'HT206B1-U10': '/diskmnt/primary/CODEX/HTAN/20230914_BRCA_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U10__20230914.ome.tiff',
#                 'HT206B1-U18': '/diskmnt/primary/CODEX/HTAN/20230914_BRCA_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U18__20230914.ome.tiff',
#                 'HT206B1-U5': '/diskmnt/primary/CODEX/HTAN/20231002_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U5__20231002.ome.tiff',
#                 'HT206B1-U13': '/diskmnt/primary/CODEX/HTAN/20231002_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U13__20231002.ome.tiff',
#                 'HT206B1-U21': '/diskmnt/primary/CODEX/HTAN/20231002_HT206B1-H2L1-2__HT206B1-H2L1-10__HT206B1-H2L1-18/HT206B1-H2L1-U21__20231002.ome.tiff',
#             }
#         }
#     },
#     'S18-9906': {
#         'order': [
#             'S18-9906-U1',
#             'S18-9906-U2',
#             'S18-9906-U3',
#             'S18-9906-U9',
#             'S18-9906-U10',
#             'S18-9906-U16',
#             'S18-9906-U17',
#             'S18-9906-U18',
#             'S18-9906-U24',
#             'S18-9906-U25',
#         ],
#         'data': {
#             'xenium': {
#                 'S18-9906-U2': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010784__S18-9906-B27Us1_2Q1__20230912__220421',
#                 'S18-9906-U17': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010784__S18-9906-B27Us1_17Q1__20230912__220421',
#                 'S18-9906-U9': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010787__S18-9906-B27Us1_9Q1__20230912__220421',
#                 'S18-9906-U24': '/diskmnt/primary/Xenium/data/20230912__220334__24002/output-XETG00122__0010787__S18-9906-B27Us1_24Q1__20230912__220421',
                
#             },
#             'multiplex': {
#                 'S18-9906-U3': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U3__20231006.ome.tiff',
#                 'S18-9906-U10': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U10__20231006.ome.tiff',
#                 'S18-9906-U18': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U18__20231006.ome.tiff',
#                 'S18-9906-U25': '/diskmnt/primary/CODEX/HTAN/20231006_Prostate_Serial_S18-9906_slide_3/S18-9906-U25__20231006.ome.tiff',
#             },
#             'visium': {
#                 'S18-9906-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/S18-9906/B27Us1_1/S18-9906-B27Us1_1Bp1/outs',
#                 'S18-9906-U16': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_FFPE/Human/S18-9906/B27Us1_16/S18-9906-B27Us1_16Bp1/outs',
#             },
#             'he': {
# #                 'S18-9906-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/images/all/A1_S18-9906-B27Us1_1.tif',
#                 'S18-9906-U1': '/diskmnt/Projects/Users/estorrs/sandbox/S18-9906_U1_tmp.tif',
# #                 'S18-9906-U16': '/diskmnt/Datasets/Spatial_Transcriptomics/images/all/B1_S18-9906-B27Us1_16.tif'
#                 'S18-9906-U16': '/diskmnt/Projects/Users/estorrs/sandbox/S18-9906_U16_tmp.tif',
#             }
#         }
#     },
    'HT413C1_Th1k4A1': {
        'order': [
            'HT413C1_Th1k4A1_U14', # U14, U18, U19, and U20 are actually in front of U1 for this sample
            'HT413C1_Th1k4A1_U18',
            'HT413C1_Th1k4A1_U19',
            'HT413C1_Th1k4A1_U20',
            'HT413C1_Th1k4A1_U1',
            'HT413C1_Th1k4A1_U2',
            'HT413C1_Th1k4A1_U3',
            'HT413C1_Th1k4A1_U4',
            'HT413C1_Th1k4A1_U7',
            'HT413C1_Th1k4A1_U8',
            'HT413C1_Th1k4A1_U9',
#             'HT413C1_Th1k4A1_U10',
            'HT413C1_Th1k4A1_U11',
            'HT413C1_Th1k4A1_U21',
            'HT413C1_Th1k4A1_U24',
            'HT413C1_Th1k4A1_U25',
            'HT413C1_Th1k4A1_U26',
            'HT413C1_Th1k4A1_U27',
            'HT413C1_Th1k4A1_U29',
            'HT413C1_Th1k4A1_U30',
            'HT413C1_Th1k4A1_U31',
            'HT413C1_Th1k4A1_U32',
            'HT413C1_Th1k4A1_U34',
            'HT413C1_Th1k4A1_U35',
            'HT413C1_Th1k4A1_U36',
            'HT413C1_Th1k4A1_U37',
            'HT413C1_Th1k4A1_U38',
            'HT413C1_Th1k4A1_U40',
            'HT413C1_Th1k4A1_U41',
            'HT413C1_Th1k4A1_U42',
            
        ],
        'data': {
            'xenium': {
                'HT413C1_Th1k4A1_U2': '/diskmnt/primary/Xenium_primary/data/20240116__200025__24019/output-XETG00122__0010369__HT413C1-Th1K4A1Us1_2__20240116__200059',
                'HT413C1_Th1k4A1_U9': '/diskmnt/primary/Xenium_primary/data/20240116__200025__24019/output-XETG00122__0010378__HT413C1-Th1K4A1Us1_9__20240116__200059',
                'HT413C1_Th1k4A1_U19': '/diskmnt/primary/Xenium_primary/data/20240116__200025__24019/output-XETG00122__0010369__HT413C1-Th1K4A1Us1_19__20240116__200059',
                'HT413C1_Th1k4A1_U25': '/diskmnt/primary/Xenium_primary/data/20240116__200025__24019/output-XETG00122__0010378__HT413C1-Th1K4A1Us1_25__20240116__200059',
                'HT413C1_Th1k4A1_U31': '/diskmnt/primary/Xenium_primary/data/20240116__200025__24019/output-XETG00122__0010369__HT413C1-Th1K4A1Us1_31__20240116__200059',
                'HT413C1_Th1k4A1_U36': '/diskmnt/primary/Xenium_primary/data/20240116__200025__24019/output-XETG00122__0010378__HT413C1-Th1K4A1Us1_36__20240116__200059',
            },
            'multiplex': { # U10 is cut off on top, U20 is in front of U1
#                 'HT413C1_Th1k4A1_U10': '/diskmnt/primary/CODEX/HTAN/20240111_Human_mCRC_serial_sectrion_HT413C1_Th1k4A1_Slide_8/HT413C1_Th1k4A1_U10__20240111.ome.tiff',
                'HT413C1_Th1k4A1_U26': '/diskmnt/primary/CODEX/HTAN/20240111_Human_mCRC_serial_sectrion_HT413C1_Th1k4A1_Slide_8/HT413C1_Th1k4A1_U26__20240111.ome.tiff',
                'HT413C1_Th1k4A1_U37': '/diskmnt/primary/CODEX/HTAN/20240111_Human_mCRC_serial_sectrion_HT413C1_Th1k4A1_Slide_8/HT413C1_Th1k4A1_U37__20240111.ome.tiff',
                'HT413C1_Th1k4A1_U42': '/diskmnt/primary/CODEX/HTAN/20240111_Human_mCRC_serial_sectrion_HT413C1_Th1k4A1_Slide_8/HT413C1_Th1k4A1_U42__20240111.ome.tiff',
                'HT413C1_Th1k4A1_U3': '/diskmnt/primary/CODEX/HTAN/20240110_Human_mCRC_Serial_section_HT413C1_Th1k4A1_Slide3/HT413C1_Th1k4A1_U3__20240110.ome.tiff',
                'HT413C1_Th1k4A1_U20': '/diskmnt/primary/CODEX/HTAN/20240110_Human_mCRC_Serial_section_HT413C1_Th1k4A1_Slide3/HT413C1_Th1k4A1_U20__20240110.ome.tiff',
                'HT413C1_Th1k4A1_U32': '/diskmnt/primary/CODEX/HTAN/20240110_Human_mCRC_Serial_section_HT413C1_Th1k4A1_Slide3/HT413C1_Th1k4A1_U32__20240110.ome.tiff',
                'HT413C1_Th1k4A1_U41': '/diskmnt/primary/CODEX/HTAN/20240110_Human_mCRC_Serial_section_HT413C1_Th1k4A1_Slide3/HT413C1_Th1k4A1_U41__20240110.ome.tiff',
            },
            'he': {
                'HT413C1_Th1k4A1_U11': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U11.ome.tif',
                'HT413C1_Th1k4A1_U14': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U14.ome.tif',
                'HT413C1_Th1k4A1_U18': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U18.ome.tif',
                'HT413C1_Th1k4A1_U1': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U1.ome.tif',
                'HT413C1_Th1k4A1_U21': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U21.ome.tif',
                'HT413C1_Th1k4A1_U24': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U24.ome.tif',
                'HT413C1_Th1k4A1_U27': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U27.ome.tif',
                'HT413C1_Th1k4A1_U29': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U29.ome.tif',
                'HT413C1_Th1k4A1_U30': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U30.ome.tif',
                'HT413C1_Th1k4A1_U35': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U35.ome.tif',
                'HT413C1_Th1k4A1_U38': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U38.ome.tif',
                'HT413C1_Th1k4A1_U40': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U40.ome.tif',
                'HT413C1_Th1k4A1_U4': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U4.ome.tif',
                'HT413C1_Th1k4A1_U8': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/he/HT413C1/HT413C1-Th1K4A1-U8.ome.tif',
            },
            'cosmx': {
                'HT413C1_Th1k4A1_U7': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/cosmx/HT413C1-Th1K4A1Us7_1.h5ad',
                'HT413C1_Th1k4A1_U34': '/diskmnt/Projects/Users/estorrs/imaging-analysis/data/htan_talk/cosmx/HT413C1-Th1K4A1Us34_1.h5ad',
            }
        }
    },
#     'HT225C1': {
#         'order': [
# #             'HT225C1-U1',
#             'HT225C1-U2',
#             'HT225C1-U3',
#             'HT225C1-U4',
#             'HT225C1-U5',
#         ],
#         'data': {
#             'visium': {
# #                 'HT225C1-U1': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U1Z1B1/outs',
#                 'HT225C1-U2': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U2Z1B1/outs',
#                 'HT225C1-U3': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U3Z1B1/outs',
#                 'HT225C1-U4': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U4Z1B1/outs',
#                 'HT225C1-U5': '/diskmnt/Datasets/Spatial_Transcriptomics/outputs_OCT/Human/HT225C1/H1/HT225C1-Th1Fc1U5Z1B1/outs',
#             },
#         }
#     },
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
                adata = xenium.adata_from_xenium(filepath)
                
                d = next(iter(adata.uns['spatial'].values()))
                
                x = d['images']['hires']
                sf = scale / d['scalefactors']['tissue_hires_scalef']
                x = rescale(rearrange(x, 'h w -> h w 1'), scale=sf)
                x = x.astype(np.float32) / x.max()
                x *= 255.
                x = x.astype(np.uint8)
                                
                tifffile.imwrite(os.path.join(output_dir, f's{idx}.tif'), x, compression='LZW')
                
            if dtype == 'cosmx':
                adata = cosmx.adata_from_cosmx(filepath)
                
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
                registered_pixels_per_micron = xenium.adata_from_xenium(v[sample]).uns['ppm']
#                 adata = adata_from_xenium(v[sample])
#                 scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
#                 registered_pixels_per_micron = scalefactors['tissue_hires_scalef'] # when read in coords are in microns, so hires_scalef is ppm
            elif k == 'multiplex':
                registered_pixels_per_micron = multiplex.pixels_per_micron(v[sample])
            elif k == 'visium':
                registered_pixels_per_micron = visium.pixels_per_micron(v[sample])
            elif k == 'cosmx':
                registered_pixels_per_micron = cosmx.adata_from_xenium(v[sample]).uns['ppm']
            elif k == 'he':
                # for now will just register to the first xenium image
                adata = xenium.adata_from_xenium(next(iter(d['data']['xenium'].values())))
                registered_pixels_per_micron = adata.uns['ppm']
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
list(zip(metadata['HT413C1_Th1k4A1']['ids'], metadata['HT413C1_Th1k4A1']['order']))

# %%

