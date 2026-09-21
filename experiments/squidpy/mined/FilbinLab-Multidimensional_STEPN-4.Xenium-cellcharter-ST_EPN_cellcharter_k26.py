# mined from: https://github.com/FilbinLab/Multidimensional_STEPN/blob/72650ca32d106fcbba0bf303a7b0720925a27dee/4.Xenium/cellcharter/ST_EPN_cellcharter_k26.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

# %%
path = "analysis/adata_obj/"
fig_path = 'figures/scatter_plots_k60/'
adata = sc.read_h5ad(path + "adata.h5ad")
batch_key = "SampleName"
cc_clust = 'cluster_cellcharter'

# %%
sq.gr.spatial_neighbors(adata, library_key=batch_key, coord_type='generic', delaunay=True, spatial_key='spatial', percentile=99)

# %%
cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter', sample_key=batch_key)

# %%
adata.write(path + "adata.h5ad", compression="gzip")

# %%
gmm = cc.tl.Cluster(
    n_clusters=26, 
    random_state=12345
)
gmm.fit(adata, use_rep='X_cellcharter')

# %%
adata.obs['cluster_cellcharter'] = gmm.predict(adata, use_rep='X_cellcharter')

# %%
# Save model
import pickle

filename = path+'cc_res/cc.pkl'
with open(filename, 'wb') as file:
    pickle.dump(autok, file)

# %%
# cell metadata
adata.obs.to_csv(path+"metadata.csv")

# %%
# anndata
adata.write(path + "adata.h5ad", compression="gzip")
