# mined from: https://github.com/fmicompbio/adv_singlecell_2022/blob/3fdf4653918852dc3cb714ca12c11a923f655a0e/day3_spatial_transcriptomics/docker/squidpy_tissuumaps.ipynb
# symbols: squidpy.datasets.visium_fluo_adata_crop, squidpy.datasets.visium_fluo_image_crop

# %%
import squidpy as sq

from spatial_analysis_toolkit.view import tissuumaps_notebook_viewer

# %%
# load the pre-processed dataset
img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

# %%
tissuumaps_notebook_viewer(adata=adata, image_container=img, image_path="test3.tif")

# %%

