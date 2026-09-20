# mined from: https://github.com/kevinyamauchi/ome-ngff-tables-prototype/blob/b4446960ec9d5ce87f2f175d70df7ea95e3ba516/notebooks/adata_visium_omezarr.ipynb
# symbols: squidpy.datasets.visium_fluo_adata_crop, squidpy.datasets.visium_fluo_image_crop

# %%
import squidpy as sq
from ngff_tables_prototype.writer import write_spatial_anndata
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData

# %load_ext autoreload
# %autoreload 2
# %load_ext lab_black

# %%
adata = sq.datasets.visium_fluo_adata_crop()
img = sq.datasets.visium_fluo_image_crop()

lib_id = "V1_Adult_Mouse_Brain_Coronal_Section_2"
spatial_key = "spatial"

# %%
plt.imshow(adata.uns[spatial_key][lib_id]["images"]["hires"])

# %%
image = adata.uns[spatial_key][lib_id]["images"]["hires"]
scalef = adata.uns[spatial_key][lib_id]["scalefactors"]["tissue_hires_scalef"]
adata.obsm[spatial_key] = adata.obsm[spatial_key] * scalef

# %%
fig, ax = plt.subplots(1, 1)
ax.imshow(image)
coords = adata.obsm[spatial_key]
ax.scatter(coords[:, 0], coords[:, 1])

# %%
adata.X = adata.X.A.copy()

# %%
tables_adata = adata.copy()
circles_adata = AnnData(None, obs = adata.obs.copy(),var=adata.var.copy(), obsm={"spatial":adata.obsm["spatial"]}, uns=adata.uns.copy())
circles_adata.obs_names = adata.obs_names.copy()
circles_adata.var_names = adata.var_names.copy()

# %%
write_spatial_anndata(
    file_path="test_visium.zarr",
    image_axes=["c", "y", "x"],
    image=np.swapaxes(adata.uns[spatial_key][lib_id]["images"]["hires"], 2, 0),
    tables_adata=tables_adata,
    circles_adata=circles_adata,
)
