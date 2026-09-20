# mined from: https://github.com/pscedu/spatial-omics-workshop-2025/blob/1ec1ab96da1ba31c7ab227bec7f0964afbb027cd/day_4_5/2.sparrow_pipeline.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import harpy as hp

# %%
import tempfile
from harpy.datasets.registry import get_registry

unit_testing = False

path = None
#path = "/staging/leuven/stg_00143/spatial_data_training" # e.g. on HPC

# The dataset will downloaded from the registry. If path is set to None, example data will be downloaded in the default cache folder of your os. Change path to your directory of choice to overwrite this behaviour.
registry = get_registry(path = path) # on Windows, set path (e.g. to r"c:\tmp")
path_image = registry.fetch("transcriptomics/resolve/mouse/20272_slide1_A1-1_DAPI.tiff")
path_coordinates = registry.fetch("transcriptomics/resolve/mouse/20272_slide1_A1-1_results.txt")

# The OUTPUT_DIR is the directory where the SpatialData .zarr will be saved. Change it to your output directory of choice.
OUTPUT_DIR =  tempfile.gettempdir()

# e.g. on HPC
#OUTPUT_DIR = "/staging/leuven/stg_00143/spatial_data_training/output_dir"

# %%
from dask_image.imread import imread

# The DAPI image is read using dask image

img=imread( path_image )

# We print the image dimensions
print('Image dimensions: ', img.shape)

img

# %%
import os
import uuid
from spatialdata import SpatialData, read_zarr

# Create an empty SpatialData object
sdata = SpatialData()

# Set the path for the SpatialData .zarr
zarr_path = os.path.join(OUTPUT_DIR, f"sdata_{uuid.uuid4()}.zarr")

# Write the SpatialData to Zarr
sdata.write(zarr_path)

# %%
# Reload the Zarr data back as a SpatialData
sdata = read_zarr(sdata.path)

# Check if SpatialData is backed (i.e. stored on disk)
sdata.is_backed()

# %%
# We add the DAPI image to the SpatialData object
sdata = hp.im.add_image_layer(
    sdata, # The SpatialData object to which the new image layer will be added.
    arr = img, # The array containing the image data to be added.
    dims = ( "c", "y", "x" ), # A tuple specifying the dimensions of the image data
    output_layer = "raw_image", # The name of the output layer where the image data will be stored.
    overwrite = True,
)

# %%
# We can access the DAPI image like this:
sdata["raw_image"] # Or, alternatively: sdata.images["raw_image"]

# %%
# Plot a crop of the DAPI image
hp.pl.plot_image(
    sdata, 
    img_layer = "raw_image" , 
    crd = [0, 6432, 0, 6432], # The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is plotted.
    figsize = (5,5),
)

# %%
# Or, alternatively, via spatialdata-plot:
import spatialdata_plot
sdata.pl.render_images("raw_image").pl.show()

# %%
# from napari_spatialdata import Interactive

# Interactive(sdata)

# %%
# Performing tiling correction
sdata, flatfields = hp.im.tiling_correction(
    sdata = sdata,
    img_layer = "raw_image",
    tile_size = 2144, # This is set to 2144 by default
    output_layer = "tiling_correction",
    crd = None,
    overwrite=True,
)

# %%
# Plot the raw and corrected image side-by-side
hp.pl.plot_image(sdata, img_layer=[ "raw_image", "tiling_correction" ], crd = [2000, 6000, 2000, 6000], figsize=(10,10))

# %%
# Perform min max filtering
sdata = hp.im.min_max_filtering(
    sdata,
    img_layer = "tiling_correction",
    output_layer = "min_max_filtered",
    size_min_max_filter = 45,
    overwrite = True,
)

# Plot the min max filtered image
hp.pl.plot_image(
    sdata,
    img_layer = "min_max_filtered",
    crd = [2000,6000,2000,6000],
    figsize = (5, 5),
)

# Perform contrast enhancement using CLAHE
sdata = hp.im.enhance_contrast(
    sdata,
    img_layer = "min_max_filtered",
    output_layer = "clahe",
    contrast_clip = 3.5,
    chunks = 20000,
    overwrite = True
)

# Plot the contrast enhanced image
hp.pl.plot_image(
    sdata,
    img_layer = "clahe",
    crd = [2000,6000,2000,6000],
    figsize = (5, 5),
)

# %%
#Interactive(sdata)

# %%
import numpy as np
from numpy.typing import NDArray

# Define your custom function
def _my_dummy_function(image: NDArray, parameter: int | float )->NDArray:
    # input (1,1,y,x)
    # output (1,1,y,x)
    print(f"Type of the image is: {type(image)}")
    print(image.shape)
    return image*parameter

fn_kwargs = {"parameter": 2}

# Apply custom function
sdata = hp.im.map_image(
    sdata,
    func = _my_dummy_function,
    fn_kwargs = fn_kwargs,
    img_layer = "raw_image",
    output_layer="dummy_image",
    chunks = 5000,
    blockwise = True, # if blockwise == True --> input to _my_dummy_function is a numpy array of size chunks, else it is a Dask array (with chunksize chunks)
    depth = 1000, # if blockwise == True, and depth specified, will use map_overlap instead of map_blocks for distributed processing
    overwrite = True,
    dtype = np.uint16,
    meta = np.array((), dtype=np.uint16),
)

# %%
from harpy.image._image import _get_spatial_element

_get_spatial_element(sdata, layer="raw_image").data.compute()[ :, :10, :10 ]

# %%
_get_spatial_element(sdata, layer="dummy_image").data.compute()[ :, :10,:10 ]

# %%
# first we rechunk on disk
from spatialdata.transformations import get_transformation

sdata=hp.im.add_image_layer(
    sdata,
    arr=sdata[ "clahe" ].data.rechunk( 2048 ),
    transformations=get_transformation( sdata[ "clahe" ], get_all=True ),
    output_layer = "clahe",
    overwrite=True,
     )

# %%
"""
ADVANCED: You can set up a local Dask distributed cluster for parallel computing. Once the cluster is created, a Dask Client is used to connect to it. 
The Dask dashboard link allows you to monitor cluster performance and task progress.
"""

from dask.distributed import Client, LocalCluster

# # Create a local Dask cluster
cluster = LocalCluster(
     n_workers=8,              # Number of worker processes
     threads_per_worker=1,    # Number of threads per worker
     memory_limit="32GB",      # Memory limit per worker
 )

# # Connect a Client to the cluster
client = Client(cluster)

# # Print the Dask dashboard link
print(client.dashboard_link)

# %%
import torch
from cellpose import models
from harpy.image import cellpose_callable

gpu = False
device = "cpu"  # mps broken in cellpose (macOS), see https://github.com/MouseLand/cellpose/issues/1063

# Perform nucleus segmentation
sdata = hp.im.segment(
    sdata,
    img_layer="clahe", # The image layer in sdata to be segmented.
    chunks=2048, #settings chunks=None would be equivalent to settings chunks=2048, as chunks on disk are 2048
    depth=200,
    model=cellpose_callable,
    # parameters that will be passed to the callable _cellpose:
    pretrained_model="nuclei", # can also be "cyto", "cyto3", or a path to a fine-tuned cellpose model.
    device=device,
    diameter=50,
    flow_threshold=0.9,
    cellprob_threshold=-4,
    output_labels_layer="segmentation_mask",
    output_shapes_layer="segmentation_mask_boundaries",
    crd=[ 2000,4000,2000,4000 ] if unit_testing else None,  # region to segment [x_min, xmax, y_min, y_max],
    overwrite=True,
)

client.close() # ADVANCED: Uncomment this when using the Dask Client.

# %%
# Plot segmentation results
hp.pl.plot_shapes(sdata, img_layer="clahe", shapes_layer="segmentation_mask_boundaries", figsize=(5,5), crd = [2000, 4000, 2000, 4000])

# %%
# or via spatialdata-plot
sdata.pl.render_images("clahe").pl.render_labels("segmentation_mask").pl.show()

# %%
# To only visualize a crop using spatialdata-plot, we can't pass any coordinates, so but we can perform a bounding box query, and then plot the resulting `SpatialData` object.
sdata_small = sdata.query.bounding_box(
    min_coordinate=[2000, 2000], max_coordinate=[4000, 4000], axes=("x", "y"), target_coordinate_system="global"
)

sdata_small.pl.render_images("clahe").pl.render_labels("segmentation_mask", fill_alpha=0.5  ).pl.show()

# %%
# Expand labels layer masks
sdata = hp.im.expand_labels_layer(
    sdata,
    labels_layer="segmentation_mask",
    distance=10, # Number of pixels to expand
    output_labels_layer="segmentation_mask_expanded", # Creates a new labels layer
    output_shapes_layer="segmentation_mask_expanded_boundaries", # Creates a new shapes layer
    overwrite=True,
)

# %%
# Plot nuclei masks vs expanded nuclei masks
hp.pl.plot_shapes(
    sdata,
    img_layer="clahe",
    shapes_layer=["segmentation_mask_boundaries", "segmentation_mask_expanded_boundaries"],
    figsize=(10,10),
    crd=[2000, 4000, 2000, 4000],
)

# %%
# Read in RESOLVE transcript data as a points layer
sdata = hp.io.read_resolve_transcripts(
    sdata, 
    output_layer="transcripts", # Name of the points layer of the SpatialData object to which the transcripts will be added.
    path_count_matrix=path_coordinates, # Path to the file containing the transcripts information specific to Resolve.
    overwrite=True
)

# Allocate transcripts to cells based on the segmentation masks
sdata = hp.tb.allocate(
    sdata=sdata,
    labels_layer="segmentation_mask", # The labels layer (i.e. segmentation mask) in `sdata` to be used to allocate the transcripts to cells.
    points_layer="transcripts", # The points layer in `sdata` that contains the transcripts.
    output_layer="table_transcriptomics", # The table layer in `sdata` in which to save the AnnData object with the transcripts counts per cell.
    update_shapes_layers=False,
    overwrite=True,
)

# %%
# Inspect the new points layer
print(type(sdata.points["transcripts"]))
sdata.points["transcripts"].head()

# %%
# Inspect the new table layer
display(sdata.tables["table_transcriptomics"])

print('Number of cells: ', len(sdata.tables["table_transcriptomics"].obs.index))
print('Number of genes: ', len(sdata.tables["table_transcriptomics"].var.index))

# %%
# Inspect the count matrix in the new table layer
sdata.tables["table_transcriptomics"].to_df().head() # On large count matrices, calls to .to_df() should be avoided

# %%
# Inspect the var of the new table layer
sdata.tables["table_transcriptomics"].var.head()

# %%
# Inspect the obs of the new table layer
sdata.tables["table_transcriptomics"].obs.head()

# %%
# Inspect the spatial coordinates stored in obsm
sdata.tables["table_transcriptomics"].obsm['spatial'][:5] # x,y,(z) coordinates of cell centre (calculated based on mean transcripts location)

# %%
# Inspect the spatialdata_attrs in .uns to check the instance_key and region_key
sdata.tables["table_transcriptomics"].uns['spatialdata_attrs']

# NOTE: The AnnData object that is added as a table layer is annotated by the labels layer "segmentation_mask". The instance_key ('cell_ID') matches the labels in "segmentation_mask".
# NOTE: Tables of a SpatialData object can be theoretically be annotated by a labels layer, a shapes layer or a points layer, but tables generated by the Harpy pipeline will always use a labels layer.

# %%
import dask.array as da

print('Number of cells in table: ', len(sdata.tables["table_transcriptomics"].obs))
print('Number of segmentation masks in labels layer: ', len(da.unique(sdata.labels["segmentation_mask"].data).compute()) - 1) # We subtract 1 because 0 is also a value, but this corresponds to the background.
print('Number of segmentation boundaries in shapes layer: ', len(sdata.shapes["segmentation_mask_boundaries"]))

# NOTE: Not all segmentation masks are included in the table layer "table_transcriptomics". This is because not all cells could be assigned transcripts.

# %%
# Interactive(sdata)

# %%
# Plot the expression of the Axl gene using hp.pl.polt_shapes()
hp.pl.plot_shapes(
    sdata,
    img_layer="clahe",
    shapes_layer="segmentation_mask_boundaries",
    figsize=(5,5),
    crd=[2000, 4000, 2000, 4000],
    table_layer="table_transcriptomics",
    column="Axl",
)

# NOTE: In Harpy/SpatialData there is a connection between tables, shapes and labels via the region_key and the cell id, which allows us to plot a certain column of a table spatially.

# %%
# Plot the expression of the Axl gene using spatialdata-plot
# Bugged, wait for https://github.com/scverse/spatialdata-plot/pull/444
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
ax = plt.gca()

gene_name = "Axl"

sdata_small = sdata.query.bounding_box(
    min_coordinate=[2000, 2000], max_coordinate=[4000, 4000], axes=("x", "y"), target_coordinate_system="global"
)

sdata_small.pl.render_labels("segmentation_mask", color=gene_name, method="datashader", fill_alpha=0.5, table_name="table_transcriptomics").pl.show(
    coordinate_systems="global", ax=ax
)

# %%
# Explore gene expression interactively using napari-spatialdata

#Interactive(sdata)

# %%
# Interactive(sdata)

# %%
# Create transcript density image
sdata = hp.im.transcript_density(
    sdata,
    img_layer="clahe", # The layer of the SpatialData object used for determining image boundary.
    points_layer="transcripts", # The layer name that contains the transcript data points, by default "transcripts".
    output_layer="transcript_density", # The name of the output image layer
    overwrite=True,
)

# %%
# Plot transcript density
hp.pl.plot_image(sdata, img_layer = ["clahe", "transcript_density"], figsize=(10,10))

# %%
# Check number of transcripts
print('Number of transcripts in points layer: ', len(sdata.points["transcripts"]))
print('Number of transcripts assigned to cells: ', sdata.tables["table_transcriptomics"].X.sum())
print('Percentage of transcripts kept: ', ((sdata.tables["table_transcriptomics"].X.sum())/len(sdata.points["transcripts"]))*100)

# NOTE: Only a fraction of transcripts are assigned to cells.

# %%
# Check number of genes
print('Number of genes in points layer: ', sdata.points['transcripts'].compute()['gene'].nunique())
print('Number of genes found in cells: ', len(sdata.tables["table_transcriptomics"].var.index))

# NOTE: In general, we don't want to lose any genes, but this may happen if they have a low abundance.

# %%
# Check which genes are not found in cells
genes_not_found_in_cells = set(sdata.points['transcripts'].compute()['gene'].unique()) - set(sdata.tables["table_transcriptomics"].var.index)

print("Number of genes not found in cells: ", len(genes_not_found_in_cells))
print("Genes not found in cells:", genes_not_found_in_cells)

# %%
# Analyse and visualize the proportion of transcripts that could not be assigned to a cell during allocation step.

df = hp.pl.analyse_genes_left_out(
    sdata,
    labels_layer="segmentation_mask",
    table_layer="table_transcriptomics",
    points_layer="transcripts",
)

# NOTE: In general we see a downward trend. The more a gene is measured, the less it is located in cells (in ratio). 
# NOTE: The function also prints the ten genes with the highest proportion of transcripts filtered out. If a lot of these genes are markers for the same cell type, you will want to find out why this is happening (bad staining, large cell body compared to nucleus, etc.)

# %%
# Inspect analyse_genes_left_out() output table
df.sort_values(by="proportion_kept", ascending=True)

# %%
# Perform preprocessing.
sdata = hp.tb.preprocess_transcriptomics(
    sdata,
    labels_layer="segmentation_mask",
    table_layer="table_transcriptomics",
    output_layer="table_transcriptomics_preprocessed", # write results to a new slot, we could also write to the same slot (when passing overwrite==True).
    min_counts=10,
    min_cells=5,
    size_norm=True,
    highly_variable_genes=False,  # If True, will only retain highly variable genes. This can be used for transcriptome-wide methods.
    max_value_scale=10, # The maximum value to which data will be scaled
    n_comps=50, # Number of principal components to calculate.
    overwrite=True,
    update_shapes_layers=False,
)

# %%
# Inspect preprocessed table
sdata.tables[ "table_transcriptomics_preprocessed" ]

# %%
# Inspect expression values
sdata.tables["table_transcriptomics_preprocessed"].to_df().head()

# %%
# Check mean expression values per gene
sdata.tables["table_transcriptomics_preprocessed"].to_df().mean(axis=0).head() # mean ~ 0

# %%
# Check standard deviation of expression values per gene
sdata.tables["table_transcriptomics_preprocessed"].to_df().std(axis=0).head() # std ~ 1

# %%
# Check max expression value per gene
sdata.tables["table_transcriptomics_preprocessed"].to_df().max(axis=0).head() # max ~ 10

# %%
# Inspect obs of preprocessed table
sdata.tables["table_transcriptomics_preprocessed"].obs.head()

# n_genes_by_counts: The number of genes with at least 1 count in a cell
# log1p_n_genes_by_counts: log1p-transformed n_genes_by_counts
# total_counts: Total number of counts for a cell
# log1p_total_counts: log1p-transformed total_counts
# pct_counts_in_top_2_genes: The percentage of the total gene expression in each cell that comes from the top 2 most highly expressed genes in that cell
# pct_counts_in_top_5_genes: The percentage of the total gene expression in each cell that comes from the top 5 most highly expressed genes in that cell 
# n_counts: Number of counts in a cell
# shapeSize: Area of cell (in pixels)

# %%
# Check sum of transcript counts
(sdata.tables["table_transcriptomics"].to_df()).sum(axis=1).head()

# %%
# Check number of genes
(sdata.tables["table_transcriptomics"].to_df()>0).sum(axis = 1).head()

# %%
# Inspect var of preprocessed table
sdata.tables["table_transcriptomics_preprocessed"].var.head()

# n_cells_by_counts: Number of cells this gene is found in
# mean_counts: Mean counts over all cells
# log1p_mean_counts: log1p of mean_counts
# pct_drop_by_counts: Percentage of cells this gene does not appear in
# total_counts: Total number of counts for a gene
# logp_total_counts: log1p of total_counts
# n_cells: Number of cells this gene is found in
# mean:
# std:

# %%
# Plot preprocessing QC plots
hp.pl.preprocess_transcriptomics(
    sdata,
    table_layer="table_transcriptomics_preprocessed",
)

# %%
# Plot total counts
hp.pl.plot_shapes(
    sdata,
    img_layer="clahe",
    table_layer="table_transcriptomics_preprocessed",
    column="total_counts",
    shapes_layer="segmentation_mask_boundaries",
    crd=[2000, 4000, 2000, 4000],
    figsize=(8,8)
)

# %%
# Filter cells on size
sdata = hp.tb.filter_on_size(
    sdata,
    labels_layer="segmentation_mask",
    table_layer="table_transcriptomics_preprocessed",
    output_layer="table_transcriptomics_filter",
    min_size=500, # Minimum cell size
    max_size=100000, # Maximum cell size
    update_shapes_layers=False,
    overwrite=True,
)

# %%
# Check which cells have been removed
hp.pl.plot_shapes(
    sdata,
    img_layer="clahe",
    table_layer="table_transcriptomics_filter",
    column="total_counts",
    shapes_layer="segmentation_mask_boundaries",
    crd=[2000, 4000, 2000, 4000],
    figsize=(8,8)
)

# %%
# Explore results interactively

#Interactive( sdata )

# %%
sdata[ "table_transcriptomics_filter" ]

# %%
import scanpy as sc

# Leiden clustering
sdata = hp.tb.leiden(
    sdata,
    labels_layer="segmentation_mask",
    table_layer="table_transcriptomics_filter",
    output_layer="table_transcriptomics_clustered",
    calculate_umap=True,
    calculate_neighbors=True,
    n_pcs=17, # The number of principal components to use when calculating neighbors.
    n_neighbors=35, # The number of neighbors to consider when calculating neighbors.
    resolution=0.8,
    rank_genes=True,
    key_added="leiden",
    overwrite=True,
)

# Plot UMAP
sc.pl.umap(sdata.tables["table_transcriptomics_clustered"], color=["leiden"], show=True)

# %%
sc.pl.rank_genes_groups(sdata.tables["table_transcriptomics_clustered"], n_genes=8, sharey=False, show=True)

# %%
# Plot clusters spatially
hp.pl.plot_shapes(
    sdata,
    img_layer="clahe",
    table_layer="table_transcriptomics_clustered",
    column="leiden",
    shapes_layer="segmentation_mask_boundaries",
    alpha=1.0,
    linewidth=0,
    crd=[2000, 4000, 2000, 4000]
)

# %%
#from napari_spatialdata import Interactive

#del sdata.tables["table_transcriptomics_clustered"].uns["leiden_colors"]
#Interactive(sdata)

# %%
import matplotlib.pyplot as plt

# for fun, also plot via spatialdataplot
plt.figure(figsize=(5, 5))
ax = plt.gca()

column = "leiden"

adata = sdata.tables[ "table_transcriptomics_clustered" ]

#cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
#                    "new_map",
#                    adata.uns[column + "_colors"],
#                    N=len(adata.uns[column + "_colors"]),
#                )

sdata_small = sdata.query.bounding_box(
    min_coordinate=[2000, 2000], max_coordinate=[4000, 4000], axes=("x", "y"), target_coordinate_system="global"
)

sdata_small.pl.render_labels("segmentation_mask", color=column, cmap=None, method="datashader", fill_alpha=1, table_name= "table_transcriptomics_clustered").pl.show(
    coordinate_systems="global", ax=ax
)

# %%
import pandas as pd

# Download annotation file from registry
path_mg = registry.fetch("transcriptomics/resolve/mouse/markerGeneListMartinNoLow.csv")

df = pd.read_csv(path_mg, index_col=0, delimiter=",")
df.columns = df.columns.str.replace(' ', '_', regex=False) # whitespaces no longer allowed since spatialdata>=0.3.0

# Inspect annotation file containing markers
display(df.head()) # This is one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column.

# %%
# Annotate cells
sdata, celltypes_scored, celltypes_all = hp.tb.score_genes(
    sdata,
    labels_layer="segmentation_mask",
    table_layer="table_transcriptomics_clustered",
    output_layer="table_transcriptomics_score_genes",
    path_marker_genes=df, # path_marker_genes can also be a dataframe
    overwrite=True,
)

# %%
# Inspect new table layer
sdata["table_transcriptomics_score_genes"]

# %%
# Inspect new table layer obs
sdata.tables["table_transcriptomics_score_genes"].obs.head()

# %%
# Plot cell type annotations on UMAP
sc.pl.umap(sdata.tables["table_transcriptomics_score_genes"], color="annotation")

# %%
# Plot cell type annotations spatially
hp.pl.plot_shapes(
    sdata,
    column="annotation",
    img_layer="clahe",
    table_layer= "table_transcriptomics_score_genes",
    shapes_layer="segmentation_mask_boundaries",
    linewidth=0,
    alpha=0.7,
    crd=None,
)

# %%
# Try calculating spatial neighbors using Squidpy
import squidpy as sq

sq.gr.spatial_neighbors(
    adata=sdata["table_transcriptomics_score_genes"], 
    coord_type="generic", # Set to 'generic' for targeted spatial transcriptomics
    n_neighs=6, # Only used when delaunay = False
    radius=None, # To compute the neighbors based on the radius
    delaunay=False, # Whether to compute the graph from Delaunay triangulation
)

sdata.tables["table_transcriptomics_score_genes"]

# %%
# BUT, this is not yet backed to the zarr store!
from spatialdata import read_zarr

sdata = read_zarr(sdata.path)

sdata.tables["table_transcriptomics_score_genes"]

# NOTE: .uns["spatial_neighbors"], .obsp["spatial_connectivities"] and .obsp["spatial_distances"] are no longer in table!

# %%
# Let's try calculating the spatial neighbors again, but we'll make sure the new table is backed to the zarr store by using hp.tb.add_table_layer().
from harpy.utils._keys import _REGION_KEY

sq.gr.spatial_neighbors(
    adata=sdata["table_transcriptomics_score_genes"], 
    coord_type="generic", # Set to 'generic' for targeted spatial transcriptomics
    n_neighs=6, # Only used when delaunay = False
    radius=None, # To compute the neighbors based on the radius
    delaunay=False, # Whether to compute the graph from Delaunay triangulation
)

region = sdata["table_transcriptomics_score_genes"].obs[_REGION_KEY].cat.categories.to_list()

sdata = hp.tb.add_table_layer(
    sdata,
    adata=sdata.tables["table_transcriptomics_score_genes"],
    output_layer="table_transcriptomics_squidpy",
    region=region, # A list of regions to associate with the table data. Typically this is all unique elements in adata.obs[_REGION_KEY].
    overwrite=True,
)

# %%
# Inspect spatial connectivities of first 10 rows and colomns
sdata.tables['table_transcriptomics_squidpy'].obsp['spatial_connectivities'].toarray()[0:10,0:10]

# %%
# Inspect number of neighbors (for first 10 cells)
sdata.tables['table_transcriptomics_squidpy'].obsp['spatial_connectivities'].toarray().sum(axis=1)[0:10]

# NOTE: Every cell has exactly 6 neighbors when using n_neigh=6

# %%
# Inspect for every cell how many cells have it as a neighbor (for first 10 cells)
sdata.tables['table_transcriptomics_squidpy'].obsp['spatial_connectivities'].toarray().sum(axis=0)[0:10]

# NOTE: Not every cell is a neighbor of exactly 6 cells when using n_neigh=6

# %%
import matplotlib.pyplot as plt

# Access the spatial connectivities matrix
matrix = sdata.tables['table_transcriptomics_squidpy'].obsp['spatial_connectivities']

# Create the plot
plt.figure(figsize=(12, 10), dpi=300)
plt.imshow(matrix.toarray(), cmap='gray_r')
plt.colorbar()
plt.title("Spatial Connectivities", fontsize=18)
plt.show()

# %%
# Inspect spatial distances
sdata.tables['table_transcriptomics_squidpy'].obsp['spatial_distances'].toarray()[0:4,0:4]

# %%
# Calculate neighborhood enrichment
sdata = hp.tb.nhood_enrichment(
    sdata, 
    labels_layer="segmentation_mask", 
    table_layer="table_transcriptomics_squidpy", 
    output_layer="table_transcriptomics_squidpy", 
    celltype_column = "annotation",
    overwrite=True
)

print(f'Filter out {(sdata.tables["table_transcriptomics_squidpy"].uns[ "annotation_nhood_enrichment" ][ "zscore" ] > 10000).sum()} outliers')
temp=sdata.tables["table_transcriptomics_squidpy"].uns[ "annotation_nhood_enrichment" ][ "zscore" ]
temp[ temp>10000 ] =0
sdata.tables["table_transcriptomics_squidpy"].uns[ "annotation_nhood_enrichment" ][ "zscore" ] = temp

# Plot neighborhood enrichment
hp.pl.nhood_enrichment(
    sdata, 
    table_layer="table_transcriptomics_squidpy",
)

# Add table layer to back to zarr
sdata = hp.tb.add_table_layer(
    sdata,
    adata=sdata.tables["table_transcriptomics_squidpy"],
    output_layer="table_transcriptomics_squidpy",
    region=region,
    overwrite=True,
)

# %%
# Calculate Moran’s I global spatial auto-correlation statistics
sq.gr.spatial_autocorr(
    adata=sdata.tables["table_transcriptomics_squidpy"],
    mode="moran",
    n_perms=100,
    n_jobs=1,
)

# Add table layer to back to zarr
sdata = hp.tb.add_table_layer(
    sdata,
    adata=sdata.tables["table_transcriptomics_squidpy"],
    output_layer="table_transcriptomics_squidpy",
    region=region,
    overwrite=True,
)

# %%
# Inspect highest Moran's I scores
sdata.tables["table_transcriptomics_squidpy"].uns["moranI"].head(10)

# %%
# Inspect lowest Moran's I scores
sdata.tables["table_transcriptomics_squidpy"].uns["moranI"].tail(10)

# %%
#Interactive(sdata)

# %%
# Region annotation in Napari

# from napari_spatialdata import Interactive
# Interactive(sdata)

# NOTE: - In napari, create a new shapes layer, annotate a region of interest, save to sdata using Shift + E and close Napari.
#       - For more info, see: https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/napari_rois.html 
#       - Currently, there is only support for saving rectangles, polygons and points.

# %%
# Let's check the sdata object to see whether the layer was correctly added
sdata

# %%
hp.pl.plot_shapes( sdata, img_layer = "clahe", shapes_layer="segmentation_mask_boundaries", crd = [ 2000, 4000, 2000, 4000 ], figsize=(5,5) )

# %%
# this cell only for unit tests to pass
from spatialdata.models import ShapesModel

if unit_testing:
    import geopandas as gpd
    from shapely.geometry import box

    # Define the rectangle boundaries
    x_min, y_min, x_max, y_max = 2250, 2250, 3000, 3000

    # Create a Shapely box (rectangle)
    rectangle = box(x_min, y_min, x_max, y_max)

    # Create a GeoDataFrame
    polygons= gpd.GeoDataFrame({'geometry': [rectangle]})
    polygons=ShapesModel.parse( polygons )
    sdata.shapes[ "region_annotation" ] = polygons
    sdata.write_element( element_name="region_annotation" )

# %%
if not unit_testing:
    # We need to make sure the shapes layer is backed to zarr
    sdata.write_element(element_name='region_annotation')
    sdata = read_zarr(sdata.path)

# %%
# Note that we can also import a GeoJSON from another source (e.g. QuPath)

# import geopandas as gpd
# gdf_regions = gpd.read_file(path_to_GeoJSON)
# sdata = hp.sh.add_shapes_layer(sdata, input=gdf_regions, output_layer='region_annotation', overwrite=True)

# %%
from shapely.geometry import Point

# Get spatial coordinates
spatial_coords = sdata.tables['table_transcriptomics_squidpy'].obsm["spatial"]
spatial_coords_df = pd.DataFrame(spatial_coords, columns=["x", "y"], index=sdata.tables['table_transcriptomics_squidpy'].obs.index)

# Define function to assign region annotations to cells
def assign_region(centroid, gdf):
    for index, row in gdf.iterrows():
        if Point(centroid).within(row["geometry"]):
            return "Yes"
    return "No"  

# Create new column in obs to check if cells are in region
sdata.tables['table_transcriptomics_squidpy'].obs["in_region"] = spatial_coords_df.apply(lambda row: assign_region((row["x"], row["y"]), sdata.shapes["region_annotation"]), axis=1)

# Add table layer to back to zarr
sdata = hp.tb.add_table_layer(
    sdata,
    adata=sdata.tables["table_transcriptomics_squidpy"],
    output_layer="table_transcriptomics_squidpy",
    region=region,
    overwrite=True,
)

# %%
# Check obs
sdata.tables['table_transcriptomics_squidpy'].obs.head()

# %%
# Plot region annotation shapes layer
hp.pl.plot_shapes(
    sdata,
    img_layer='clahe',
    shapes_layer='region_annotation',
    alpha=0.5,
    crd=[2000, 4000, 2000, 4000]
)

# %%
# Plot cells colored according to in_region column
hp.pl.plot_shapes(
    sdata,
    column="in_region",
    img_layer="clahe",
    table_layer="table_transcriptomics_squidpy",
    shapes_layer="segmentation_mask_boundaries",
    linewidth=0,
    alpha=0.7,
    cmap="rainbow",
    crd=[2000, 4000, 2000, 4000]
)

# %%
# Export image as tiff
from skimage.io import imsave
import numpy as np

if not unit_testing:

    # Save AnnData as h5ad
    sdata.tables["table_transcriptomics_squidpy"].write(os.path.join(OUTPUT_DIR, 'adata.h5ad'))

    # Export shapes layer as GeoJSON
    sdata.shapes['region_annotation'].to_file(os.path.join(OUTPUT_DIR, "region_annotation.geojson"), driver="GeoJSON")


    img = sdata.images['clahe'].data.compute()
    imsave(os.path.join(OUTPUT_DIR, "clahe.tiff"), img)
