# mined from: https://github.com/kevinyamauchi/ome-ngff-tables-prototype/blob/b4446960ec9d5ce87f2f175d70df7ea95e3ba516/notebooks/adata_segment_omezarr.ipynb
# symbols: squidpy.datasets.mibitof, squidpy.pl.spatial_segment

# %%
import squidpy as sq
from ngff_tables_prototype.writer import write_spatial_anndata
import numpy as np

# %load_ext autoreload
# %autoreload 2
# %load_ext lab_black

# %%
adata = sq.datasets.mibitof()
lib_id = "point8"
spatial_key = "spatial"
adata = adata[adata.obs.library_id == lib_id].copy()
adata.uns[spatial_key][lib_id]["images"].keys()

# %%
adata.uns[spatial_key][lib_id]["images"]["hires"].shape

# %%
image = adata.uns[spatial_key][lib_id]["images"]["hires"]
segment = adata.uns[spatial_key][lib_id]["images"]["segmentation"]

# %%
sq.pl.spatial_segment(
    adata,
    seg_cell_id="cell_id",
    library_key="library_id",
    color="Cluster",
    library_id="point8",
)

# %%
adata.X = adata.X.A.copy()
tables_adata = adata.copy()

# %%
write_spatial_anndata(
    file_path="test_segment.zarr",
    image_axes=["c", "y", "x"],
    image=np.swapaxes(image, 2, 0),
    label_image=segment,
    tables_adata=tables_adata,
    tables_instance_key="cell_id",
)
