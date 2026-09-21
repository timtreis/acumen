# mined from: https://github.com/arjunrajlaboratory/melanoma_geomx_public/blob/b7d3ab029bcdc2c8eff1d58759f78c0a03e652d6/Paper/plotScripts/spatialGenomics/SG_PDX3_plotting.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import matplotlib.pyplot as plt

from copy import deepcopy

import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
from anndata import AnnData
import random

# %%
pdx3 = sc.read_h5ad('../../extractedData/SG_PDX3_processed.h5ad')

# %%
pdx3

# %%
sc.set_figure_params(dpi=300, dpi_save=300, color_map = 'magma_r')

# %%
sc.pl.umap(pdx3, color='leiden', add_outline=True, legend_loc='on data',
               legend_fontsize=8, legend_fontoutline=2,frameon=False,
               title='UMAP with leiden clusters', palette='Set1', save = 'umap_leiden.svg')

# %%
sc.pl.pca(pdx3, color = 'MLANA', add_outline=True, frameon=False, save = 'pca_MLANA.svg')

# %%
sc.pl.pca(pdx3, color = 'leiden', add_outline=True, frameon=False, legend_loc='on data',
               legend_fontsize=8, legend_fontoutline=2, palette='Set1', groups = ['1', '3'], save = 'pca_leiden13.svg')

# %%
sc.pl.rank_genes_groups_dotplot(pdx3, n_genes = 2, vmax = 3, save = 'dotplot_2genes.svg')

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="leiden", size=5, library_id="spatial", figsize=(10, 10), palette='Set1', frameon=False, save = 'full_leiden_spatial.svg'
)

# %%
pdx3.uns['leiden_colors']

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="leiden", size=2, library_id="spatial", figsize=(10, 10), frameon=False, groups = ['1', '3'],
save = 'leiden13_spatial.svg')

# %%
pdx3.obs['is_leiden_6'] = [True if six == '6' else False for six in pdx3.obs['leiden']]

# %%
pdx3.obs['is_leiden_7'] = [True if seven == '7' else False for seven in pdx3.obs['leiden']]

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="is_leiden_6", size=1, library_id="spatial", figsize=(10, 10), frameon=False,
save = 'leiden6_spatial.svg'
)

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="is_leiden_7", size=1, library_id="spatial", figsize=(10, 10), frameon=False,
save = 'leiden7_spatial.svg'
)

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="LOXL2", size=2, library_id="spatial", figsize=(10, 10), frameon=False, cmap = 'Reds',
save = 'LOXL2_spatial.svg'
)

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="DDIT4", size=2, library_id="spatial", figsize=(10, 10), frameon=False, cmap = 'Reds',
save = 'DDIT4_spatial.svg'
)

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="VEGFA", size=2, library_id="spatial", figsize=(10, 10), frameon=False, cmap = 'Reds',
save = 'VEGFA_spatial.svg'
)

# %%
NGFR_threshold = 3

NGFR_values = pdx3[:, pdx3.var_names == 'NGFR'].X
NGFR_values = NGFR_values.flatten()

# Check if each value is above the threshold
NGFR_is_above_threshold = NGFR_values >= NGFR_threshold

pdx3.obs['NGFR_high'] = NGFR_is_above_threshold

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="NGFR_high", size=1, library_id="spatial", figsize=(10, 10), frameon=False, cmap = 'magma_r',
#save = 'VEGFA_spatial.svg'
)

# %%
sq.pl.spatial_scatter(
    pdx3, shape=None, color="NGFR_high", size=1, library_id="spatial", figsize=(10, 10), frameon=False, palette = 'magma_r',
save = 'NGFR_high_spatial.svg'
)

# %%
sc.pl.violin(pdx3, ['NGFR', 'COL9A3', 'S100B'], groupby = 'leiden', save = 'NGFR_COL9A3_S100B_leiden_violin.svg')

# %%
sc.pl.rank_genes_groups_dotplot(pdx3, n_genes = 10, vmax = 3)
