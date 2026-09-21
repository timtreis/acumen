# mined from: https://github.com/yuanzhiyuan/SODB_analysis/blob/0a2fbc31f31a9087bf93e989a843e3628a7dfebc/Raw2Anndata/SpatialProteomics/4i/gut2018multiplexed.ipynb
# symbols: squidpy.datasets.four_i

# %%
# the original paper stated that full 4i data can be downloaded from http://www.cellatlas.org/.
# But we failed to find them,
# so we instead use the data from squidpy

# %%
import squidpy as sq

# %%
adata = sq.datasets.four_i()
