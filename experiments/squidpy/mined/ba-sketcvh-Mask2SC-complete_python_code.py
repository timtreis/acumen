# mined from: https://github.com/ba-sketcvh/Mask2SC/blob/4673db2758d7b8374ce77a3dec8eefd9e536bbf5/complete_python_code.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
# 1. Define path to image, name of image path to Xenium folder,and set pixel size 
# (Must change to correct value before running)
#=====================================================================================================================
xenium_dir = "Path_of_Xenium_output" # (e.g. D:/Xenium/Sample1)
img_dir = "Path_to_Image" # (e.g. D:/Sample1/Exported_Channels)
channel_name = "Name_of_biomarker" # not the full file name, (e.g. use MCT1, but not MCT1.tiff)
pixel_size = 0.2125 # Find it in Qupath

# %%
# 2. Read in image information and cell boundary information (Run without change)
#=====================================================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from imageio.v2 import imread
from shapely.geometry import Polygon
from rasterio.features import rasterize
from affine import Affine
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Join the directory and file name to get the full path
img_path = os.path.join(img_dir, f'{channel_name}.tiff')
# Read the image
img = np.asarray(imread(img_path).astype(np.float32))
# Join the directory and file name to get the full path
boundary_path = os.path.join(xenium_dir, "cell_boundaries.csv.gz")
# Read the CSV file
boundary_data = pd.read_csv(boundary_path)

# %%
# 3. Assign pixel to cells and calculated mean intensity of pixels in each cell (Run without change)
#====================================================================================================
height, width = img.shape
boundary_data["vertex_x"] = boundary_data["vertex_x"] / pixel_size
boundary_data["vertex_y"] = boundary_data["vertex_y"] / pixel_size
def make_valid_polygon(df):
    coords = list(zip(df["vertex_x"], df["vertex_y"]))
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = Polygon(coords)
    return poly.buffer(0)
boundary_polygons = (
    boundary_data.groupby("cell_id", group_keys=False)
    .apply(make_valid_polygon, include_groups=False)
    .reset_index()
    .rename(columns={0: "geometry"}))
boundary_gdf = gpd.GeoDataFrame(boundary_polygons, geometry="geometry")
boundary_gdf["label"] = range(1, len(boundary_gdf) + 1)
label_to_cell_id = dict(zip(boundary_gdf["label"], boundary_gdf["cell_id"]))
transform = Affine.translation(0, height) * Affine.scale(1, -1)
cell_mask = rasterize(
    [(geom, label) for geom, label in zip(boundary_gdf.geometry, boundary_gdf.label)],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.int32)
unique_ids = np.unique(cell_mask)
print("Unique values in rasterized mask:", unique_ids)
cell_mask_flipped = np.flipud(cell_mask)
valid_pixels = cell_mask_flipped > 0
cell_ids = cell_mask_flipped[valid_pixels]
pixel_values = img[valid_pixels]
mipc = pd.DataFrame({"label": cell_ids, "intensity": pixel_values})
mipc = mipc.groupby("label").mean().reset_index()
mipc.rename(columns={"intensity": channel_name}, inplace=True)
mipc["cell_id"] = mipc["label"].map(label_to_cell_id)
mipc = mipc[["cell_id", channel_name]]

# %%
# 4. Validate overlay (Optional, run without change)
#====================================================================================================
print("Rendering polygon overlay on image for validation...")
print("May take a few minutes...")
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# First subplot: original image
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Image Only")
axes[0].axis("off")

# Second subplot: image with boundary overlay
axes[1].imshow(img, cmap='gray')
boundary_gdf.plot(ax=axes[1], facecolor='none', edgecolor='red', linewidth=0.1)
axes[1].set_title("Image with Boundary")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# %%
# 5. Integrate MFI information to Xenium adata (Run without change)
#==========================================================================================================================

# Construct full paths
h5_path = os.path.join(xenium_dir, "cell_feature_matrix.h5")
cells_csv_path = os.path.join(xenium_dir, "cells.csv.gz")

# Read the files
adata = sc.read_10x_h5(h5_path)
cells_df = pd.read_csv(cells_csv_path, index_col="cell_id")

#integrate MFI information to xenium adata
cells_df = cells_df[cells_df.index.isin(adata.obs_names)]
adata.obsm["spatial"] = cells_df[["x_centroid", "y_centroid"]].values
mipc = mipc.set_index('cell_id')
mipc = mipc.reindex(adata.obs_names)
adata.uns["spatial"] = {
        "library_id": {
            "images": {"hires": None, "lowres": None},
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "spot_diameter_fullres": 1.0
            }
        }
    }
for col in mipc.columns:
    adata.obs[col] = mipc[col]
adata = adata[adata.obs.sort_values(channel_name, ascending=True).index]

# %%
# 6. Plotting (Run without change)
#====================================================================================================
fig, ax = plt.subplots(figsize=(6, 6))
sq.pl.spatial_scatter(
    adata,
    color= channel_name,
    size=20,
    cmap='viridis',
    ax=ax,
    img=False,
    vmin=0, # you may change the value to threshold the displayed image
    vmax=100, # you may change the value to threshold the displayed image
    figsize=(10,10)
)
ax.set_aspect('equal', adjustable='datalim')
ax.set_title(channel_name)
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()
