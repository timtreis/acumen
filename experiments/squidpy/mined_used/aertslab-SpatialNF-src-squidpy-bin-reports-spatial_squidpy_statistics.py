# mined from: https://github.com/aertslab/SpatialNF/blob/8db7eaadc774812e0fc320f55d564df04298e2f9/src/squidpy/bin/reports/spatial_squidpy_statistics.ipynb
# symbols: squidpy.pl.centrality_scores, squidpy.pl.co_occurrence, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.ripley

# %%
# Import packages
import scanpy as sc
import squidpy as sq

# %%
# plot settings
sc.set_figure_params(dpi=150, fontsize=10, dpi_save=600)

# %%
adata = sc.read_h5ad(filename=FILE)
KEY = annotations_to_plot[0]

# %%
if KEY+'_nhood_enrichment' in adata.uns:
    sq.pl.nhood_enrichment(adata, cluster_key=KEY, method='ward', mode='count')
else:
    print(f"Neighborhood enrichment using {KEY} have not been computed")

# %%
if KEY+'_interactions' in adata.uns:
    sq.pl.interaction_matrix(adata, cluster_key=KEY, method='ward')
else:
    print(f"Interaction matrix using {KEY} have not been computed")

# %%
if KEY+'_co_occurrence' in adata.uns:
    for k in adata.obs.loc[:, KEY].unique():
        sq.pl.co_occurrence(adata, KEY, clusters=[k])
else:
    print(f"Cluster co-occurences using {KEY} have not been computed")

# %%
if KEY+'_centrality_scores' in adata.uns:
    sq.pl.centrality_scores(adata, KEY, figsize=(15,5))
else:
    print(f"Centrality scores using {KEY} have not been computed")

# %%
if KEY + '_ripley_F' in adata.uns:
    sq.pl.ripley(adata, cluster_key=KEY, mode='F')
else:
    print(f"Ripley F statistic using {KEY} has not been computed")    
if KEY + '_ripley_G' in adata.uns:
    sq.pl.ripley(adata, cluster_key=KEY, mode='G')
else:
    print(f"Ripley G statistic using {KEY} has not been computed")
if KEY + '_ripley_L' in adata.uns:
    sq.pl.ripley(adata, cluster_key=KEY, mode='L')
else:
    print(f"Ripley L statistic using {KEY} has not been computed")
