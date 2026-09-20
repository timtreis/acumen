# mined from: https://github.com/standardgalactic/thesis-work/blob/76c9fbde4da54112bd06301b14fe78ef4f2879de/Squidpy/Visium HE/Visium HE tutorial.py
# symbols: squidpy.im.calculate_image_features

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:22:38 2022

@author: Mark Zaidi

Squidpy documentation:
    https://squidpy.readthedocs.io/en/latest/api.html
    https://squidpy.readthedocs.io/en/latest/auto_tutorials/tutorial_visium_hne.html
    Also check scanpy as a lot of the IO stuff came from that
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#OKAY, START FROM SCRATCH. Mixing and matching functions from different parts of the squidpy and scanpy tutorial is
clearly going to cause issues. Instead, follow the tutorial of squidpy that show how to load in your own dataset. DO NOT
DO IT ANY DIFFERENTLY.    
    
    
    
    
    
"""
##% Import libraries
import scanpy as sc
import anndata as ad
import squidpy as sq

import numpy as np
import pandas as pd

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")
#%% Load data
# load the pre-processed dataset
# img = sq.datasets.visium_hne_image()
# adata = sq.datasets.visium_hne_adata()
# Load one of our datasets
data_dir=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\50D'

#adata will also contain the visium image. We need to put the image into an im.ImageContainer object with the appropriate scale factor
#adata = sq.read.visium(path=data_dir,counts_file='filtered_feature_bc_matrix.h5')
#Need to use scanpy's import function, if we intend to use some of scanpy's visualization tools
adata = sc.read_visium(path=data_dir,count_file='filtered_feature_bc_matrix.h5',source_image_path=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\50D\spatial\tissue_hires_image.png' )

#Dataset name (a.k.a. library?) will be the only key in the dict
dataset_name = list(adata.uns['spatial'].keys())[0]
#Extract the high res Visium images scale factor, ignoring the low res one
HE_res_scale=adata.uns['spatial'][dataset_name]['scalefactors']['tissue_hires_scalef']
#Create an image container object from the high res
#img=sq.im.ImageContainer.from_adata(adata=adata,img_key='hires_image',scale=HE_res_scale)
#Scanpy labels the high resolution image as hires, whereas squidpy is hires_image
img=sq.im.ImageContainer.from_adata(adata=adata,img_key='hires',scale=HE_res_scale)

#%% Calculate image features
# calculate features for different scales (higher value means more context)
for scale in [1.0, 2.0]:
    feature_name = f"features_summary_scale{scale}"
    sq.im.calculate_image_features(
        adata,
        img,
        features="summary",
        key_added=feature_name,
        scale=scale,
        n_jobs = 30,
        show_progress_bar=False #YOU NEED THIS ELSE IT WILL ERROR
    )
#%% Preview image
sc.pl.spatial(adata,color='in_tissue')
#%% helper function returning a clustering
def cluster_features(features: pd.DataFrame, like=None) -> pd.Series:
    """
    Calculate leiden clustering of features.

    Specify filter of features using `like`.
    """
    # filter features
    if like is not None:
        features = features.filter(like=like)
    # create temporary adata to calculate the clustering
    adata = ad.AnnData(features)
    # important - feature values are not scaled, so need to scale them before PCA
    sc.pp.scale(adata)
    # calculate leiden clustering
    sc.pp.pca(adata, n_comps=min(10, features.shape[1] - 1))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)

    return adata.obs["leiden"]


# calculate feature clusters
adata.obs["features_cluster"] = cluster_features(adata.obsm["features"], like="summary")

# compare feature and gene clusters
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.pl.spatial(adata, color=["features_cluster", "cluster"])