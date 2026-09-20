# mined from: https://github.com/abyssum/spatial/blob/8d0f5e8eeefeda47a2960cd4a6cabda863a18c13/Mass_cyto/mass_cyto_data.ipynb
# symbols: squidpy.datasets.imc, squidpy.gr.centrality_scores, squidpy.gr.co_occurrence, squidpy.gr.interaction_matrix, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.centrality_scores, squidpy.pl.co_occurrence, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
# %matplotlib inline

# %%
import squidpy as sq

print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
adata = sq.datasets.imc()

# %%
sq.pl.spatial_scatter(adata, shape=None, color="cell type", size=10)

# %%
sq.gr.co_occurrence(adata, cluster_key="cell type")
sq.pl.co_occurrence(
    adata,
    cluster_key="cell type",
    clusters=["basal CK tumor cell", "T cells"],
    figsize=(15, 4),
)

# %%
# Neighborhood enrichment

# %%
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="cell type")
sq.pl.nhood_enrichment(adata, cluster_key="cell type")

# %%
sq.gr.interaction_matrix(adata, cluster_key="cell type")
sq.pl.interaction_matrix(adata, cluster_key="cell type")

# %%
sq.gr.centrality_scores(
    adata,
    cluster_key="cell type",
)
sq.pl.centrality_scores(adata, cluster_key="cell type", figsize=(20, 5), s=500)

# %%

