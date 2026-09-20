# mined from: https://github.com/kevinyamauchi/spatial-analysis-toolkit/blob/8a6352219a17d5b7b62fc3129ea556a4f39d617b/examples/squidpy_tissuumaps.ipynb
# symbols: squidpy.datasets.visium_fluo_adata_crop, squidpy.datasets.visium_fluo_image_crop

# %%
import squidpy as sq


from spatial_analysis_toolkit.view import tissuumaps_notebook_viewer

# %%
# load the pre-processed dataset
img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

# %%
tissuumaps_notebook_viewer(adata=adata, image_container=img, port=7000)

# %%

