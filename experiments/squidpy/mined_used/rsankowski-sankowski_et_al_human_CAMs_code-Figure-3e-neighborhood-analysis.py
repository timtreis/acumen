# mined from: https://github.com/rsankowski/sankowski_et_al_human_CAMs_code/blob/5b942af73680023b740abb2f9a44bd33dd4f57dc/Figure-3e-neighborhood-analysis.ipynb
# symbols: squidpy.gr.interaction_matrix, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment

# %%
import scanpy as sc
import squidpy as sq
import pandas as pd

#sc.settings.verbosity = 3adata = sq.datasets.visium_hne_adata()
sc.settings.set_figure_params(dpi=80, facecolor="white")

# %%
counts =  pd.read_csv('data/Out_CtrlCtx4_Scale50_High_Prior_s50_sd50_conf08/segmentation_counts_filt_prior08.csv',
                     index_col=0) # feature matrix
coordinates = pd.read_csv('data/Out_CtrlCtx4_Scale50_High_Prior_s50_sd50_conf08/b1hi_ctrl_coord_prior08.csv',
                    index_col=0)  # spatial coordinates
#image = NULL  # image
labels = pd.read_csv('data/predictions_ctx4.csv',
                    index_col=0)

# %%
counts2 = counts.transpose()

# %%
adata = sc.AnnData(counts2.loc[labels.index], obsm={"spatial": coordinates.loc[labels.index],
                             "celltype" : labels})

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
adata

# %%
sq.gr.spatial_neighbors(adata)

# %%
## add celltype as clustering variable
adata.obs["celltype"] = list(labels['label'])
adata.obs["celltype"] = adata.obs["celltype"].astype('category')                           

# %%
sq.gr.nhood_enrichment(adata, cluster_key="celltype")

# %%
adata.uns["celltype_nhood_enrichment"]

# %%
sq.pl.nhood_enrichment(adata, 
                       cluster_key="celltype", 
                       method="average", 
                       figsize=(9, 9),
                        save='cartana_celltype_neighborhood_enrichment.pdf')

# %%
sq.gr.interaction_matrix(adata, cluster_key="celltype")

# %%
adata.uns["celltype_interactions"]

# %%
sq.pl.interaction_matrix(adata, cluster_key="celltype", method="average", figsize=(5, 5))

# %%
adata_ctx = adata[adata.obs['celltype'].isin(['Oligo', 'CAMs', 'Neuron','Astro','MG','Mural cells','Endothelial cells'])]
## run interaction analysis
sq.gr.interaction_matrix(adata_ctx, cluster_key="celltype")

# %%
sq.pl.interaction_matrix(adata_ctx, 
                         cluster_key="celltype", 
                         method="average", 
                         figsize=(5, 5),
                        save='ctx_celltype_interactions.svg')

# %%
adata_lm = adata[adata.obs['celltype'].isin(['CAMs', 'Lymphocytes','Fibroblast','Mural cells','Endothelial cells'])]
## run interaction analysis
sq.gr.interaction_matrix(adata_lm, cluster_key="celltype")

# %%
sq.pl.interaction_matrix(adata_lm, 
                         cluster_key="celltype", 
                         method="average", 
                         figsize=(9, 9),
                        save='lm_celltype_interactions.pdf')

# %%
import sys
sys.modules.keys()
import types
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__

# %%
# !pip freeze > requirements.txt

# %%

