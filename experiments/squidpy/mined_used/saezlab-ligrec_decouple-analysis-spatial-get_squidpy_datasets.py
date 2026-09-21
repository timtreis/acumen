# mined from: https://github.com/saezlab/ligrec_decouple/blob/3130efb5c1616bbaf6dbc3880549cf6110c1c6f7/analysis/spatial/get_squidpy_datasets.ipynb
# symbols: squidpy.datasets.merfish, squidpy.datasets.seqfish, squidpy.datasets.slideseqv2, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment

# %%
# import packages
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np

# %%


# %%
"""""
Title: Function to return the EM and do NES with Squidpy

params:
    adata - anndata
    coord_type - type of coordinate system
    cluster_key - variable where clustering is stored
    spatial_key - where spatial coordinates are stored - for some reason relevant for neighbourhood analysis
    kwargs - passed to plot
    
    
returns: list with both adata and nes
"""""
def get_adata_nes(adata, cluster_key, spatial_key = "spatial", **kwargs):
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key=spatial_key)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
    sq.pl.nhood_enrichment(adata, cluster_key=cluster_key, **kwargs)
    
    # nes
    cats = adata.obs[cluster_key].cat.categories
    nes_key = "{}_nhood_enrichment".format(cluster_key)
    nes = pd.DataFrame(adata.uns[nes_key]['zscore'], index=cats, columns=cats)
    
    both = dict(adata = adata, nes = nes)
    return(both)

# %%


# %%
# seqFISH
adata = sq.datasets.seqfish()
get_adata_nes(adata, cluster_key="celltype_mapped_refined", method="ward")

# %%


# %%
# merFISH
adata = sq.datasets.merfish()
seq_both = get_adata_nes(adata, cluster_key="Cell_class", spatial_key="spatial3d", method="single", cmap="inferno", vmin=-50, vmax=100)

# %%


# %%


# %%
adata = sq.datasets.slideseqv2()

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.nhood_enrichment(adata, cluster_key="cluster")

# %%
adata.obsm['deconvolution_results']

# %%


# %%
adata = sq.datasets.seqfish()

# %%
adata

# %%
sc.pl.spatial(adata, color="celltype_mapped_refined", spot_size=0.03)

# %%


# %%
help(sq.gr.spatial_neighbors)

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.nhood_enrichment(adata, cluster_key="celltype_mapped_refined")
sq.pl.nhood_enrichment(adata, cluster_key="celltype_mapped_refined", method="ward")

# %%


# %%
nn = sq.gr.nhood_enrichment(adata, cluster_key="celltype_mapped_refined", copy = True)

# %%


# %%
cats = adata.obs["celltype_mapped_refined"].cat.categories

# %%
nes = pd.DataFrame(adata.uns['celltype_mapped_refined_nhood_enrichment']['zscore'],index=cats,columns=cats)

# %%
nes

# %%
help(sq.pl.nhood_enrichment)

# %%

nes

# %%


# %%
adata = sq.datasets.merfish()

# %%
adata

# %%
sc.pl.spatial(adata, color="Cell_class", spot_size=0.03)

# %%


# %%
sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial3d")
sq.gr.nhood_enrichment(adata, cluster_key="Cell_class")

# %%
sq.pl.nhood_enrichment(adata, cluster_key="Cell_class", method="single", cmap="inferno", vmin=-50, vmax=100)

# %%
cluster_key = "Cell_class"
cats = adata.obs[cluster_key].cat.categories
nes_key = "{}_nhood_enrichment".format(cluster_key)

# %%
nes = pd.DataFrame(adata.uns[nes_key]['zscore'],index=cats,columns=cats)

# %%
nes = pd.DataFrame(adata.uns['celltype_mapped_refined_nhood_enrichment']['zscore'],index=cats,columns=cats)

# %%
nes

# %%
dict(adata = adata,
     nes = nes)

# %%


# %%


# %%

