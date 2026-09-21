# mined from: https://github.com/dieterich-lab/ScNaST/blob/06c35f848a8ac93321b1fb57b2d8f9dcff08d15f/paper/figures/2/figure2.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment

# %%
import os
import pandas as pd
import numpy as np

import scanpy as sc
import squidpy as sq

# %%
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import colors

# from Scrublet
def darken_cmap(cmap, scale_factor):
    cdat = np.zeros((cmap.N, 4))
    for ii in range(cdat.shape[0]):
        curcol = cmap(ii)
        cdat[ii,0] = curcol[0] * scale_factor
        cdat[ii,1] = curcol[1] * scale_factor
        cdat[ii,2] = curcol[2] * scale_factor
        cdat[ii,3] = 1
    cmap = cmap.from_list(cmap.N, cdat)
    return cmap
cmaps = darken_cmap(plt.cm.Reds, 0.9)

bone_reversed = plt.cm.get_cmap('bone_r')

# %%
parent = '/prj/Florian_Leuschner_spatial/analysis/Nanopore/' # change this path!

# Illumina and Nanopore results 
# use local objects, we'll copy them to the repository in due time/provide them as supplement

illumina = sc.read_h5ad('/prj/Florian_Leuschner_spatial/analysis/Illumina/data/anatomical_regions_and_cell_props_bbknn.h5ad')
illumina
nanopore = sc.read_h5ad('/prj/Florian_Leuschner_spatial/analysis/Nanopore/Nanopore/data/anatomical_regions_and_cell_props_scnast.h5ad')
nanopore

# %%
for library in ['A', 'B', 'C', 'D']:
    adata = nanopore[nanopore.obs.library_id == library, :].copy()
    sq.gr.co_occurrence(adata, cluster_key="anatomical_region")
    filen = os.path.join(parent, 'ScNaST', 'paper', 'figures', '2', f'cooccurance-{library}.svg')
    sq.pl.co_occurrence(adata, cluster_key="anatomical_region", clusters="Infarct", save=filen)
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="anatomical_region")
    filen = os.path.join(parent, 'ScNaST', 'paper', 'figures', '2', f'nhoodenrichment-{library}.svg')
    sq.pl.nhood_enrichment(adata, cluster_key="anatomical_region", cmap=bone_reversed, title='', annotate=True, figsize=(4,4), save=filen)
    

# %%

