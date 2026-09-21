# mined from: https://github.com/zhanglabtools/NPC_scRNA_Spatial_code/blob/3c167b70aa4a485d149719f04cb3f61b7d91c002/Spatial_transcriptomics/06_Neighbor_enrichment.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment

# %%
import numpy as np
import pandas as pd
import os
import anndata as ad
import scanpy as sc
import squidpy as sq

# %%
files = os.listdir('E:\\spatial interactions\\data_store\\')

# %%
for file in files:
    adata = sc.read_h5ad('E:\\spatial interactions\\data_store\\'+file)
    adata = adata[~adata.obs['leiden2'].isin(['others'])]
    adata = adata[~adata.obs['leiden2'].isin(['other'])]
    leiden_list = list(adata.obs['leiden2'])
    leiden_list = ['Malignant' if x == 'malignant' else x for x in leiden_list]
    adata.obs['leiden2'] = leiden_list
    adata.obs['leiden2'] = adata.obs['leiden2'].astype('category')
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="leiden2")
    sq.pl.nhood_enrichment(adata, cluster_key="leiden2")
    pd.DataFrame(adata.uns['leiden2_nhood_enrichment']['zscore'], index =list(adata.obs['leiden2'].cat.categories), columns=list(adata.obs['leiden2'].cat.categories)).to_csv('./Neighbor Enrichment/'+file[0:-5]+'.csv' )

# %%

