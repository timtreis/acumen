# mined from: https://github.com/Super-LzzZ/CellNiche/blob/af58974ded7cf57299a9f8952d4cc6dffee39c6f/tutorial/mulitSlice.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
# ! hostname

# %%
# enable autoreload
# %load_ext autoreload
# %autoreload 2

# %%
import os
import time
import sys
import random
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad

from sklearn import metrics
import multiprocessing as mp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import scipy.sparse as sp
import scipy.linalg

import warnings
warnings.filterwarnings("ignore")

# %%
plt.rcParams['pdf.fonttype'] = 42
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.set_figure_params(dpi=150,
                     dpi_save=300,
#                      facecolor='w',
#                      frameon=False, # frameon=True
#                      figsize=(4,4)
                    ) 
# %config InlineBackend.figure_format='retina'
# %matplotlib inline

# %%
# import cellniche as cn
sys.path.append('/share/home/liangzhongming/phd_code/ModelTest/CellNiche')
import cellniche as cn

# %%
def format_gene_names(genes):
    return [gene.upper() if len(gene) == 1 else gene[0].upper() + gene[1:].lower() for gene in genes]

atlas_dirs = [
    '/share/home/liangzhongming/phd_code/530/ProcessedData/MouseBrain/Atlas1',
    '/share/home/liangzhongming/phd_code/530/ProcessedData/MouseBrain/Atlas2',
    '/share/home/liangzhongming/phd_code/530/ProcessedData/MouseBrain/Atlas3',
    '/share/home/liangzhongming/phd_code/530/ProcessedData/MouseBrain/Atlas4'
]

target_files = {
    'well11_cellTypt.h5ad', 
    'C57BL6J-638850.38.h5ad', 
    'Zhuang-ABCA-1.079.h5ad', 
    'S2R1_cellTypt.h5ad'
}

selected_files = {}
for atlas_dir in atlas_dirs:
    h5ad_files = [f for f in os.listdir(atlas_dir) if f.endswith('.h5ad')]
    matched = [f for f in h5ad_files if f in target_files]
    if matched:
        selected_files[atlas_dir] = [os.path.join(atlas_dir, m) for m in matched]

print("The files to be processed：")
for k, v in selected_files.items():
    print(f"{os.path.basename(k)}: {[os.path.basename(f) for f in v]}")

adata_list = []
section_ids = []


for atlas_dir, files in selected_files.items():
    atlas_name = os.path.basename(atlas_dir)
    
    for file_path in files:

        sample = sc.read_h5ad(file_path)
        

        if 'Atlas1' in atlas_name or 'Atlas4' in atlas_name:
            sample.var_names = format_gene_names(sample.var_names)
        elif 'Atlas2' in atlas_name or 'Atlas3' in atlas_name:
            if 'gene_symbol' in sample.var.columns:
                sample.var.index = sample.var['gene_symbol']
                sample.var_names_make_unique()
        

        rename_dict = {
            'class_name': 'class',
            'subclass_name': 'subclass',
            'supertype_name': 'supertype',
            'cluster_name': 'cluster'
        }

        valid_columns = {k: v for k, v in rename_dict.items() if k in sample.obs.columns}
        if valid_columns:
            sample.obs.rename(columns=valid_columns, inplace=True)
            
        for prob_col in ['supertype_bootstrapping_probability', 'subclass_bootstrapping_probability']:
            if prob_col in sample.obs.columns:
                sample = sample[sample.obs[prob_col] >= 0.5].copy()
                

        if 'spatial' not in sample.obsm:
            x_col = next((col for col in ['x', 'X', 'center_x'] if col in sample.obs.columns), None)
            y_col = next((col for col in ['y', 'Y', 'center_y'] if col in sample.obs.columns), None)
            
            if x_col and y_col:
                sample.obsm['spatial'] = sample.obs[[x_col, y_col]].astype(float).values
                
        slice_id = os.path.splitext(os.path.basename(file_path))[0]
        sample.obs['atlas'] = atlas_name
        sample.obs['slice_id'] = slice_id
        
        adata_list.append(sample)
        section_ids.append(f"{atlas_name}_{slice_id}")

# %%
adata_concat = sc.concat(
    adata_list, 
    label="sample",
    keys=section_ids,
    join='inner', # inner outer
    index_unique='-' 
)

# %%
adata_concat

# %%
adata_concat.obs.head()

# %%


# %%
sq.gr.spatial_neighbors(
    adata_concat, 
    library_key='sample',
    coord_type='generic',
    delaunay=True,
    spatial_key='spatial', 
    percentile=99
)

# %%
adata_concat

# %%
adata_concat

# %%


# %%
conn = adata_concat.obsp["spatial_connectivities"].tocsr()
row, col = conn.nonzero()

sample_labels = adata_concat.obs["sample"].astype(str).to_numpy()
cross_edge_mask = sample_labels[row] != sample_labels[col]
n_cross_edges = int(cross_edge_mask.sum())

print(f"Number of graph edges: {len(row)}")
print(f"Number of cross-sample edges: {n_cross_edges}")

assert n_cross_edges == 0, (
    "Cross-sample edges were detected. "
    "Please check Squidpy graph construction and library_key='sample'."
)

# %%


# %%
adata_concat.write("/share/home/liangzhongming/phd_code/ModelTest/CellNiche/data/mergedAtlas1234_4slices_inner.h5ad")

# %%
sc.pp.subsample(adata_concat, fraction=0.1, random_state=0, copy=False)

# %%
adata_concat.write("/share/home/liangzhongming/phd_code/ModelTest/CellNiche/data/mergedAtlas1234_4slices_subsampled10percent.h5ad")

# %%


# %%
# ! cat ../configs/multiSlices.yaml

# %%
MultiSlices = cn.cli(["--config", "../configs/multiSlices.yaml"])

# %%
MultiSlices

# %%

