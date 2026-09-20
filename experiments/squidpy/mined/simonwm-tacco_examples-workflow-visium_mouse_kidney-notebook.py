# mined from: https://github.com/simonwm/tacco_examples/blob/ed61ddc584be72217fe83d33b8995589264efd50/workflow/visium_mouse_kidney/notebook.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import scanpy as sc

import squidpy as sq
import tacco as tc

# The notebook expects to be executed either in the workflow directory or in the repository root folder...
sys.path.insert(1, os.path.abspath('workflow' if os.path.exists('workflow/common_code.py') else '..')) 
import common_code

# %%
data_path = common_code.find_path('results/visium_mouse_kidney/data')
plot_path = common_code.find_path('results/visium_mouse_kidney')

# %%
reference = ad.read(f'{data_path}/TabulaMurisKidney.h5ad')
slide_1 = ad.read(f'{data_path}/MouseKidney_visium.h5ad')
slide_1.var_names_make_unique()
slide_2 = ad.read(f'{data_path}/MouseKidneyCoronal_v110_visium.h5ad')
slide_2.var_names_make_unique()

# %%
# Merge slides
slide_m = ad.concat([slide_1,slide_2], index_unique="_")
slide_m.obs['slide'] = slide_m.obs['slide'].astype("category")

# %%
matplotlib.rcParams['figure.dpi'] = 100.0

axsize = np.array([4,3])*0.5

# %%
# Annotate with merged slides
tc.tl.annotate(slide_m,reference,'cell_ontology_class',result_key='cell_ontology_class',multi_center=3,lamb=1e-3)

# %%
fig = tc.pl.scatter(slide_m, keys='cell_ontology_class', group_key='slide', position_key=['x','y'], joint=True, point_size=1, axsize=axsize, noticks=True, axes_labels=['X','Y'], );

# %%
# Find consistent regions on merged slides
tc.tl.find_regions(slide_m,key_added='regions',batch_key='slide',position_weight=20,resolution=1.0,annotation_key=None,amplitudes=False,cross_batch_overweight_factor=100,n_neighbors=1000)
slide_m.obs['regions'] = slide_m.obs['regions'].map(lambda x: f'region_{x}').astype('category')

# %%
fig = tc.pl.scatter(slide_m,keys='regions',group_key='slide',joint=True,axsize=axsize,point_size=1,noticks=True,axes_labels=['X','Y'],share_scaling=True);

# %%
slide_1.obs_names = slide_1.obs_names.map(lambda x: x + "_0")
slide_2.obs_names = slide_2.obs_names.map(lambda x: x + "_1")

# %%
# Map annotations back to original adatas 
for col in slide_m[slide_m.obs["slide"]=="slide_1"].obsm["cell_ontology_class"].columns:
    slide_1.obs[col] = slide_m.obsm["cell_ontology_class"][col]

for col in slide_m[slide_m.obs["slide"]=="slide_2"].obsm["cell_ontology_class"].columns:
    slide_2.obs[col] = slide_m.obsm["cell_ontology_class"][col]

slide_1.obs["regions"] = slide_m[slide_m.obs["slide"]=="slide_1"].obs["regions"]    
slide_2.obs["regions"] = slide_m[slide_m.obs["slide"]=="slide_2"].obs["regions"]    

# %%
celltypes = ['endothelial cell', 'fenestrated cell', 'fibroblast', 'kidney collecting duct cell', 'kidney tubule cell', 'leukocyte', 'macrophage', 'smooth muscle cell']
for slide in [slide_1, slide_2]:
    fig, axes = tc.pl.subplots(n_x = 4, n_y = 2);
    axes = np.ravel(axes)
    for j, celltype in enumerate(celltypes):
        sq.pl.spatial_scatter(slide, library_key='hires', color=celltype, ax=axes[j], img_alpha=0.5)
    axes = np.reshape(axes, (4,2))
    plt.show()

# %%
fig, axes = tc.pl.subplots(n_x=2, n_y=1, wspace=.5)
sq.pl.spatial_scatter(slide_1, library_key='hires', color="regions", ax=axes[0][0], img_alpha=0.5);
sq.pl.spatial_scatter(slide_2, library_key='hires', color="regions", ax=axes[0][1], img_alpha=0.5);

# %%

