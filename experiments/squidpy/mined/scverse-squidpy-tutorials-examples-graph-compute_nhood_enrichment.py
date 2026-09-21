# mined from: https://github.com/scverse/squidpy-tutorials/blob/4984ce95c7b9858ac8be8b662a32e86316d3870e/examples/graph/compute_nhood_enrichment.ipynb
# symbols: squidpy.datasets.visium_fluo_adata, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment

# %%
# %matplotlib inline

# %%
import squidpy as sq

adata = sq.datasets.visium_fluo_adata()
adata

# %%
sq.gr.spatial_neighbors(adata)

# %%
sq.gr.nhood_enrichment(adata, cluster_key="cluster")

# %%
sq.pl.nhood_enrichment(adata, cluster_key="cluster")
