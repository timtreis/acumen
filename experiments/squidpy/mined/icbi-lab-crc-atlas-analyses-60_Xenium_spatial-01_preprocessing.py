# mined from: https://github.com/icbi-lab/crc-atlas/blob/82c15ecc2ea36baa0c4cffe8818bf58e21dcfc48/analyses/60_Xenium_spatial/01_preprocessing.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import numpy as np
import pandas as pd
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import scanpy as sc
import squidpy as sq
import spatialdata as sd
import spatialdata_io as sdio
import spatialdata_plot

import torch
import scvi

import matplotlib.patches as patches
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed

from spatialdata.transformations import (
        Affine,
        Identity,
        MapAxis,
        Scale,
        Sequence,
        Translation,
        get_transformation,
        get_transformation_between_coordinate_systems,
        set_transformation,
    )

# set project dir paths
prjdir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if prjdir not in sys.path:
    sys.path.append(prjdir)

n_jobs=32
sc.settings.n_jobs=n_jobs
sc.set_figure_params(dpi=100, frameon=True, vector_friendly=True, fontsize=10)

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('grey_to_blue', ['lightgrey', 'mediumblue'])

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# %%
# %load_ext autoreload
# %autoreload 2
import src.spatial_helpers.spatial as spp
import src.spatial_helpers.spatialplot as spl
import src.spatial_helpers.sc as scp
scp.set_all_seeds()

# %%
samplesdir = '../../data/xenium/samples'
processed_datadir = '../../data/xenium/processed'
resultsdir = '../../results/xenium' # dir to save results
os.makedirs(resultsdir, exist_ok=True)
sc.settings.figdir = resultsdir

# %%
sdata = sd.read_zarr(os.path.join(processed_datadir, 'crca_xenium.zarr'))

# %%
sdata['table'].obs['transcript_density'] = sdata['table'].obs['total_counts'] / sdata['table'].obs['cell_area']

# %%
# spatially isolated cells
n_neighs=50
sq.gr.spatial_neighbors(sdata['table'], coord_type='generic', n_neighs=n_neighs)
sdata['table'].obs['mean_neighbor_dist'] = np.sum(sdata['table'].obsp['spatial_distances'], axis=1) / n_neighs

# %%
fig, axs = plt.subplots(1, 5, figsize=(15, 5))
fig.subplots_adjust(right=1.5)
sns.histplot(sdata['table'].obs, x='transcript_counts', hue = 'name', multiple='stack', log_scale=True, ax=axs[0])
sns.histplot(sdata['table'].obs, x='transcript_density', hue = 'name', multiple='stack', log_scale=True, ax=axs[1])
sns.histplot(sdata['table'].obs, x='nucleus_area', hue = 'name', multiple='stack', log_scale=True, ax=axs[2])
sns.histplot(sdata['table'].obs, x='control_probe_counts', hue = 'name', multiple='stack', log_scale=True, ax=axs[3])
sns.histplot(sdata['table'].obs, x='mean_neighbor_dist', hue = 'name', multiple='stack', log_scale=True, ax=axs[4])
fig.savefig(os.path.join(resultsdir, 'QC_histplots_sample.jpeg'), bbox_inches='tight', dpi=300)

# %%
fig, axs = plt.subplots(1, 5, figsize=(15, 5))
fig.subplots_adjust(right=1.5)
sns.histplot(sdata['table'].obs, x='transcript_counts', hue = 'segmentation_method', multiple='stack', log_scale=True, ax=axs[0])
sns.histplot(sdata['table'].obs, x='transcript_density', hue = 'segmentation_method', multiple='stack', log_scale=True, ax=axs[1])
sns.histplot(sdata['table'].obs, x='nucleus_area', hue = 'segmentation_method', multiple='stack', log_scale=True, ax=axs[2])
sns.histplot(sdata['table'].obs, x='control_probe_counts', hue = 'segmentation_method', multiple='stack', log_scale=True, ax=axs[3])
sns.histplot(sdata['table'].obs, x='mean_neighbor_dist', hue = 'segmentation_method', multiple='stack', log_scale=True, ax=axs[4])
fig.savefig(os.path.join(resultsdir, 'QC_histplots_segmentation_method.jpeg'), bbox_inches='tight', dpi=300)

# %%
# filtering
adata = sdata['table'][ (sdata['table'].obs['transcript_counts'] >= 15) & (sdata['table'].obs['control_codeword_counts'] < 2) ]

# %%
# knn spatial neighbors
adata = spp.spatial_neighbors(adata, sample_key='name', n_neighs=6, max_distance=50)
adata = spp.spatial_neighbors(adata, sample_key='name', n_neighs=15, max_distance=50)
adata = spp.filter_n_neighbors(adata, key='spatial_n6r50_connectivities', min_neighbors=4)
spp.check_adj_matrix(adata.obsp[adata.uns['spatial_n6r50_neighbors']['connectivities_key']], adata.obs, 'name')

# %%
# delaunay spatial neighbors
adata = spp.spatial_neighbors(adata, sample_key='name', delaunay=True, max_distance=50)
adata = spp.filter_n_neighbors(adata, key='delaunayr50_connectivities', min_neighbors=3)
spp.check_adj_matrix(adata.obsp[adata.uns['delaunayr50_neighbors']['connectivities_key']], adata.obs, 'name')
adata.write_h5ad(os.path.join(processed_datadir, 'crc_ffpe_filtered.h5ad'))

# %%
adata = sc.read_h5ad(os.path.join(processed_datadir, 'crc_ffpe_filtered.h5ad'))

# %%
adata.layers['counts'] = adata.X.copy()
adata.layers['norm'] = adata.X.copy()
del adata.X

# %%
sc.pp.normalize_total(adata, layer='norm', key_added='norm_factor', inplace=True)
sc.pp.log1p(adata, layer='norm')

# %%
res = [2.5, 3]
adata = scp.pp(adata, layer='norm', resolution=res)

# %%
sc.pl.embedding(adata, basis='norm_pca_nb_umap', color=['name', 'tissue_region'])

# %%
adata.write_h5ad(os.path.join(processed_datadir, 'crc_ffpe_norm.h5ad'))

# %%
adata = sc.read_h5ad(os.path.join(processed_datadir, 'crc_ffpe_norm.h5ad'))

# %%
res = [2.5, 3]
batch_key='name'

# %%
# adata = scp.run_scvi(adata, key='scvi120', n_layers=1, n_latent=20, layer='counts', batch_key='name', get_expr=True, save=os.path.join(processed_datadir, 'scvi', 'model_scvi120'))
# adata = scp.pp(adata, use_rep='scvi120', layer='norm', resolution=res)
# sc.pl.embedding(adata, basis='scvi120_nb_umap', color=['name', 'tissue_region', 'scvi120_nb_leiden_2.5'], show=False)
# plt.savefig(os.path.join(resultsdir, 'scvi120_umap.jpg'), dpi=300, bbox_inches='tight')

# %%
adata = scp.run_scvi(adata, key='scvi130', n_layers=1, n_latent=30, layer='counts', batch_key='name', get_expr=True, save=os.path.join(processed_datadir, 'scvi', 'model_scvi130'))
adata = scp.pp(adata, use_rep='scvi130', layer='norm', resolution=res)
sc.pl.embedding(adata, basis='scvi130_nb_umap', color=['name', 'tissue_region', 'scvi130_nb_leiden_2.5'], show=False)
plt.savefig(os.path.join(resultsdir, 'scvi130_umap.jpg'), dpi=300, bbox_inches='tight')

# %%
# adata = scp.run_scvi(adata, key='scvi220', n_layers=2, n_latent=20, layer='counts', batch_key='name', get_expr=False, save=os.path.join(processed_datadir, 'scvi', 'model_scvi220'))
# adata = scp.pp(adata, use_rep='scvi220', layer='norm', resolution=res)
# sc.pl.embedding(adata, basis='scvi220_nb_umap', color=['name', 'tissue_region', 'scvi220_nb_leiden_2.5'], show=False)
# plt.savefig(os.path.join(resultsdir, 'scvi220_umap.jpg'), dpi=300, bbox_inches='tight')

# %%
# adata = scp.run_scvi(adata, key='scvi230', n_layers=2, n_latent=30, layer='counts', batch_key='name', get_expr=False, save=os.path.join(processed_datadir, 'scvi', 'model_scvi230'))
# adata = scp.pp(adata, use_rep='scvi230', layer='norm', resolution=res)
# sc.pl.embedding(adata, basis='scvi230_nb_umap', color=['name', 'tissue_region', 'scvi230_nb_leiden_2.5'], show=False)
# plt.savefig(os.path.join(resultsdir, 'scvi230_umap.jpg'), dpi=300, bbox_inches='tight')

# %%
# adata = scp.run_scvi(adata, key='scvipat', n_layers=1, n_latent=30, layer='counts', batch_key='patient_id', get_expr=False, save=os.path.join(processed_datadir, 'scvi', 'model_scvipat'))
# adata = scp.pp(adata, use_rep='scvipat', layer='norm', resolution=res)
# sc.pl.embedding(adata, basis='scvipat_nb_umap', color=['name', 'tissue_region', 'scvipat_nb_leiden_2.5'], show=False)
# plt.savefig(os.path.join(resultsdir, 'scvipat_umap.jpg'), dpi=300, bbox_inches='tight')

# %%
# adata = scp.run_scvi(adata, key='scvibatch', n_layers=1, n_latent=30, layer='counts', batch_key='batch_factor', get_expr=False, save=os.path.join(processed_datadir, 'scvi', 'model_scvibatch'))
# adata = scp.pp(adata, use_rep='scvibatch', layer='norm', resolution=res)
# sc.pl.embedding(adata, basis='scvibatch_nb_umap', color=['name', 'tissue_region', 'scvibatch_nb_leiden_2.5'], show=False)
# plt.savefig(os.path.join(resultsdir, 'scvibatch_umap.jpg'), dpi=300, bbox_inches='tight')

# %%
model = scvi.model.SCVI.load(os.path.join(processed_datadir, 'scvi','model_scvi130'), adata=adata)
adata.obsm['scvi'] = model.get_latent_representation()
adata.layers['scvi'] = csr_matrix(model.get_normalized_expression(transform_batch=None))

# %%
adata = scp.pp(adata, use_rep='scvi130', layer='norm', n_iterations=5, resolution=[2.5, 3, 3.5, 3.8, 4])

# %%
adata.write_h5ad(os.path.join(processed_datadir, 'crc_ffpe_integrated.h5ad'))

# %%
sdata = sd.read_zarr(os.path.join(processed_datadir, 'crca_xenium.zarr'))
sdata['int'] = adata
sdata = spp.match_ids(sdata, ['cell_boundaries'], table_key='int')
sdata.tables['int'].obs['region'] = 'cell_boundaries'
sdata.set_table_annotates_spatialelement('int', region_key='region', region='cell_boundaries')
sdata.write_element('int', overwrite=True)
