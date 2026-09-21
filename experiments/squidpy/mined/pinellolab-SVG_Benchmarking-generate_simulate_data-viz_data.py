# mined from: https://github.com/pinellolab/SVG_Benchmarking/blob/31579000e77af4f22a075365e255026a20a80d79/generate_simulate_data/viz_data.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
import scipy as sp

# %%
input_dir = '../../results/01_generate_simulate_data'

# %%
adata = sc.read_h5ad(f'{input_dir}/01_10x_Visium_mouse_brain.h5ad')

# %%
adata

# %%
adata.var

# %%
sq.pl.spatial_scatter(adata, 
                      color=['Ttr_1', 'Ttr_0.8', 'Ttr_0.6', 'Ttr_0.4', 'Ttr_0.2', 'Ttr_0'],
                      ncols=6,
                      shape=None, size=30)

# %%
sc.pl.violin(adata, 
             keys=['Ttr_1', 'Ttr_0.8', 'Ttr_0.6', 'Ttr_0.4', 'Ttr_0.2', 'Ttr_0']
            )

# %%
sq.pl.spatial_scatter(adata, 
                      color=['S100a5_1', 'S100a5_0.8', 'S100a5_0.6', 'S100a5_0.4', 'S100a5_0.2', 'S100a5_0'],
                      ncols=6,
                      shape=None, size=30)

# %%
sc.pl.violin(adata, 
             keys=['S100a5_1', 'S100a5_0.8', 'S100a5_0.6', 'S100a5_0.4', 'S100a5_0.2', 'S100a5_0']
            )

# %%
adata = sc.read_h5ad(f'{input_dir}/02_slide_seqv2_mouse_hippocampus.h5ad')

# %%
adata.var

# %%
sq.pl.spatial_scatter(adata, 
                      color=['Mbp_1', 'Mbp_0.8', 'Mbp_0.6', 'Mbp_0.4', 'Mbp_0.2', 'Mbp_0'], 
                      ncols=6, vmax=1,
                      shape=None, size=0.1)

# %%
adata = sc.read_h5ad(f'{input_dir}/05_10x_Visium_Mouse_Embryo.h5ad')

# %%
adata.var

# %%
sq.pl.spatial_scatter(adata, 
                      color=['Fga_1', 'Fga_0.8', 'Fga_0.6', 'Fga_0.4', 'Fga_0.2', 'Fga_0',
                             'Nsg2_1', 'Nsg2_0.8', 'Nsg2_0.6', 'Nsg2_0.4', 'Nsg2_0.2', 'Nsg2_0',
                            'Sftpc_1', 'Sftpc_0.8', 'Sftpc_0.6', 'Sftpc_0.4', 'Sftpc_0.2', 'Sftpc_0'], 
                      shape=None, size=3, ncols=6,)

# %%
adata = sc.read_h5ad(f'{input_dir}/08_10x_Visium_Human_Lymph_Node.h5ad')

# %%
adata.layers['counts'].todense()

# %%
adata.var

# %%
sq.pl.spatial_scatter(adata, 
                      color=['FDCSP_1', 'FDCSP_0.8', 'FDCSP_0.6', 
                             'FDCSP_0.4', 'FDCSP_0.2', 'FDCSP_0'], 
                      shape=None, size=6)

# %%

