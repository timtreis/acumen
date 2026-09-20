# mined from: https://github.com/pinellolab/SVG_Benchmarking/blob/31579000e77af4f22a075365e255026a20a80d79/prepare_reference_data/38_MERFISH_Mouse_cortex.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.spatial_scatter

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
import scipy as sp

# %%
adata = sc.read_h5ad('../../data/from_spatial_omics/MERFISH/MERFISH_Fang2022Conservation_mouse1.AUD_TEA_VIS.242.unexpand_data.h5ad')

# %%
adata

# %%
adata.layers['counts'] = adata.X.copy()

# %%
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=50)

# %%
adata

# %%
sq.pl.spatial_scatter(adata, color=['leiden'], size=10, shape=None)

# %%
# remove MT genes
non_mito_genes_list = [name for name in adata.var_names if not name.startswith('mt-')]
adata = adata[:, non_mito_genes_list]

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(adata, mode="moran", 
                       n_perms=100, n_jobs=10, 
                       genes=adata.var_names)

# %%
n_svgs = 25
sel_genes = (
    adata.uns["moranI"]["I"].sort_values(ascending=False).head(n_svgs).index.tolist()
)

# %%
sq.pl.spatial_scatter(
    adata, color=sel_genes[:10], figsize=(5, 5), size=1, 
    cmap="Reds", shape=None, use_raw=False
)

# %%
# select top 50 variable genes as reference
adata = adata[:, sel_genes]

# %%
adata

# %%
adata.write_h5ad('../../results/00_prepare_reference_data/38_MERFISH_Mouse_cortex.h5ad')

# %%

