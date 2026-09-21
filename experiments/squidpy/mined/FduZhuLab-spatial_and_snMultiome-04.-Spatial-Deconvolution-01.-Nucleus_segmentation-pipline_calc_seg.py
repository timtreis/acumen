# mined from: https://github.com/FduZhuLab/spatial_and_snMultiome/blob/320ca72bf9356280f42d9a67fd979fcee79d155b/04. Spatial Deconvolution/01. Nucleus_segmentation/pipline_calc_seg.py
# symbols: squidpy.im.ImageContainer, squidpy.im.calculate_image_features

import os
import gc
import pickle
import argparse
import functools

import numpy as np
import pandas as pd

import scanpy as sc
import squidpy as sq

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

os.chdir(os.getenv("HOME"))

print = functools.partial(print, flush=True)

# Argements
num_cores = int(os.getenv("SLURM_CPUS_PER_TASK"))

parser = argparse.ArgumentParser()
parser.add_argument('--donor', '-d', type=str, required=True)
parser.add_argument('--region', '-r', type=str, required=True)
parser.add_argument('--spt_base', '-spt', type=str, required=True)
parser.add_argument('--img_base', '-img', type=str, required=True)
parser.add_argument('--suffix', '-suffix', nargs='?', type=str, default="")

parser.add_argument('--add_diam', '-add_diam', nargs='?', type=float, default=0.)

parser.add_argument('--n_jobs', nargs='?', type=int, default=num_cores)

args = parser.parse_args()

# Load Data
sample_dir = '/'.join([args.spt_base, args.donor, args.region])
spt_ada = sc.read('/'.join([sample_dir, "spt_adata.h5ad"]))

spt_img = sq.im.ImageContainer().load(
    '/'.join([args.img_base, args.donor, args.region, "zarr" + args.suffix]),
)
spt_ada = spt_img.subset(spt_ada)

# add spot diameter
_key = list(spt_ada.uns['spatial'].keys())[0]
orig_diam = spt_ada.uns['spatial'][_key]['scalefactors']['spot_diameter_fullres']
spt_ada.uns['spatial'][_key]['scalefactors']['spot_diameter_fullres'] += args.add_diam

# Cal Segment Features
features_kwargs = {
    "segmentation": {
        "label_layer": "segmented_stardist",
        "props": ["label", "centroid"]
    }
}

with np.errstate(divide='ignore',invalid='ignore'):
    sq.im.calculate_image_features(
        spt_ada, spt_img, layer="image", key_added="image_features",
        features_kwargs=features_kwargs, features="segmentation",
        mask_circle=True, n_jobs=args.n_jobs
    )

# save
spt_ada.uns['spatial'][_key]['scalefactors']['spot_diameter_fullres'] = orig_diam
spt_ada.write_h5ad('/'.join([sample_dir, "spt_adata.h5ad"]))

file = '/'.join([sample_dir, "image_features" + args.suffix + ".pkl"])
pickle.dump(
    spt_ada.obsm["image_features"], open(file, 'wb')
)

print("\nCalc Features Success")
