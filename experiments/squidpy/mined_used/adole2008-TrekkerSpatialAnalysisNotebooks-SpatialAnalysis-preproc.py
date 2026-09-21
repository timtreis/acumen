# mined from: https://github.com/adole2008/TrekkerSpatialAnalysisNotebooks/blob/7b9eef184e810a937290a1992b31140eebb3a543/SpatialAnalysis/preproc.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.im.ImageContainer, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
import squidpy as sq
import numpy as np
import pandas as pd

import anndata as ad 
import scanpy as sc

sc.logging.print_header()

# %%
mouse_liver_adata = sc.read_h5ad("/home/ad1829/20260622_YR_RnD_020c_U-Trekker_MOv1_TrekkerRun/TrekkerOutputData/TrekkerU_MouseLiver_ConfPositioned_anndata_matched.h5ad")
mouse_liver_adata.obsm.keys()

mouse_liver_adata.obsm["spatial"] = mouse_liver_adata.obsm["X_spatial"]
#copied X_spatial obsm to spatial because squidpy expects the spatial data 
#to be in obsm called "spatial"

# %%
mouse_liver_adata

# %%
import tifffile

img = tifffile.imread("/home/ad1829/20260622_YR_RnD_020c_U-Trekker_MOv1_TrekkerRun/GMP-YR-1417-2026, Yashika Rustagi/YR_RnD_020c_mLiver3.tif")
# print(img.shape)

# %%
import matplotlib.pyplot as plt

img_small = img[::16, ::16]
coords_small = mouse_liver_adata.obsm["spatial"]

plt.imshow(img_small)
x_coords = coords_small[:,0]/3.55
y_coords = coords_small[:,1]/3.55

x_rot = -y_coords
y_rot = x_coords

plt.scatter(
  x_rot,
  y_rot,
  s=1
)
plt.show()

# %%
mouse_liver_adata

# %%
#standard scanpy clustering and umap workflow

sc.pp.normalize_total(mouse_liver_adata)
sc.pp.log1p(mouse_liver_adata)
sc.pp.pca(mouse_liver_adata)
sc.pp.neighbors(mouse_liver_adata)
sc.tl.umap(mouse_liver_adata)
sc.tl.leiden(mouse_liver_adata)

mouse_liver_adata

# %%
#Creating a Squidpy Img container to store the scaled-down H&E image

img = sq.im.ImageContainer(img_small)
#actual image is stored in image attribute of Squidpy.ImageContainer obj.
#Can extract the image as a numpy array and create SpatialData.Image2DModel from it


sq_img = (
  img.data.image.squeeze()
  .transpose("channels","y", "x")
  .to_numpy()
)

sq_img

# %%
import spatialdata as sd 
import spatialdata_plot as sdp
import geopandas as gpd
from shapely.geometry import Point
from spatialdata.models import Image2DModel, ShapesModel, TableModel

plt.imshow(sq_img.transpose(1,2,0)) #matplotlib expects (x, y, c)

# %%
img_for_sdata = Image2DModel.parse(data=sq_img, scale_factors=(2,2,2))
img_for_sdata

# %%
mouse_liver_adata.obsm["spatial"] = np.column_stack([x_rot,y_rot])
mouse_liver_adata.obsm["spatial"]

# %%
# Creating a Points object using the spatial data
from spatialdata.models import PointsModel

centers = mouse_liver_adata.obsm["spatial"]
coords_df = pd.DataFrame(
  centers,
  columns=["x", "y"],
  index=mouse_liver_adata.obs_names
)

points_for_sdata = PointsModel.parse(coords_df)
points_for_sdata

# %%
mouse_liver_adata_for_sdata = mouse_liver_adata

mouse_liver_adata_for_sdata.uns["spatialdata_attrs"] = {
  "region" : "nuclei",
  "region_key": "region",
  "instance_key": "cell_id"
}

mouse_liver_adata_for_sdata.obs["region"] = pd.Categorical(
    ["nuclei"] * mouse_liver_adata.n_obs
)

mouse_liver_adata_for_sdata.obs["cell_id"] = mouse_liver_adata.obs_names

# %%
mouse_liver_adata_for_sdata.obs.keys()

# %%
import celltypist
from celltypist import models

print(f"TrekkerU_MouseLiver: {mouse_liver_adata_for_sdata.n_obs} cells and {mouse_liver_adata_for_sdata.n_vars} genes") 

# %%
mouse_liver_adata_for_sdata.obs

# %%
print(models.models_description())

# %%
model = models.Model.load(model = "Healthy_Mouse_Liver.pkl")
#Note: You need to download the model from celltypist.org/models
model

# %%
# mouse_liver_adata_for_sdata.X.max()
mouse_liver_adata_for_sdata.X.min()

# %%
mouse_liver_adata_for_sdata.n_vars

# %%
preds = celltypist.annotate(mouse_liver_adata_for_sdata, model = 'Healthy_Mouse_Liver.pkl', majority_voting = True)

# %%
preds.predicted_labels

# %%
adata_cell_annotated = preds.to_adata()
adata_cell_annotated.obs

# %%

fig, ax = plt.subplots(1,2, figsize=(20,20))

fig.suptitle("Celltypist Annotation UMAP, Legend on Data")

ax[0].set_title('Predicted Labels'),
predicted_label_UMAP=sc.pl.umap(adata_cell_annotated, color = ['predicted_labels'], legend_loc = 'on data', ax=ax[0], show=False) 

ax[1].set_title('Majority Voting'),
majority_vote_UMAP=sc.pl.umap(adata_cell_annotated, color = ['majority_voting'], legend_loc = 'on data', ax=ax[1], show=False)

# %%
# Subset of Fibroblasts

mouse_liver_fibroblasts = mouse_liver_adata_for_sdata[mouse_liver_adata_for_sdata.obs["majority_voting"] == 'Fibroblasts'].copy()
mouse_liver_fibroblasts

# %%
# Repeat the clustering process on only the fibroblasts
sc.pp.highly_variable_genes(mouse_liver_fibroblasts)
sc.pp.pca(mouse_liver_fibroblasts)
sc.pp.neighbors(mouse_liver_fibroblasts)
sc.tl.umap(mouse_liver_fibroblasts)
sc.tl.leiden(mouse_liver_fibroblasts,resolution=0.6)

# %%
# Marker Genes Analysis - Fibroblasts and HSCs
fibroblast_markers = {
  "Heterogenous Fibroblasts" : ["PDGFRA", "COL1A1", "COL3A1", "DPT", "MFAP4", "GPX3", "FMOD", "DPT"], 
  # ^general fibroblasts, contains all general markers
  "Capsular Fibroblasts" : ["PDGFRA", "PDGFRB", "SCARA5", "FBLN2", "ADGRD1", "OSR1"],
  "Peribiliary Fibroblasts" : ["THY1", "GFRA2", "WIF1", "NKD2"],
  # "Portal Fibroblasts" : [], - couldn't find any unique markers for this subtype
  "Myofibroblasts" : ["SAA3", "C3", "COL1A1", "S100A6", "GAS6", "ACTA2", "CCL2", "COL3A1", "COL5A2", "ID3"],
  # Used Krenkel et al for markers here, but grouped MFB1-4 as a single group (myofibroblasts)
  "Hepatic Stellate Cells" : ["NGFR", "RELN", "PDGFRA", "PDGFRB", "NCAM1", "WT1"]
  #ncam1 and wt1 are only expressed significantly in one type of HSC cells, so they aren't required to classify them
}

#did not include negative markers, maybe that is the next step to refine the analysis further

# %%
marker_genes_in_fibroblast_cluster = {}

for ct, markers in fibroblast_markers.items():
  markers_found = []
  for marker in markers:
    if marker in mouse_liver_fibroblasts.var.index:
      markers_found.append(marker)
  marker_genes_in_fibroblast_cluster[ct] = markers_found

# %%
mouse_liver_fibroblasts.layers["counts"] = mouse_liver_fibroblasts.X

# %%
sc.tl.pca(mouse_liver_fibroblasts, n_comps=50, use_highly_variable=True)

# %%
sc.pp.neighbors(mouse_liver_fibroblasts)
sc.tl.umap(mouse_liver_fibroblasts)

# %%
fibroblast_cts = [
  "Heterogenous Fibroblasts",
  "Capsular Fibroblasts",
  "Peribiliary Fibroblasts",
  "Myofibroblasts",
  "Hepatic Stellate Cells"
]

# %%
for ct in fibroblast_cts:
  print(f"{ct.upper()}:") # prints cell subtype name
  sc.pl.umap(
    mouse_liver_fibroblasts,
    color = marker_genes_in_fibroblast_cluster[ct],
    vmin = 0,
    vmax = "p99",
    sort_order = False,
    frameon = False,
    cmap = "Reds"
  )
  print("\n\n\n\n")
  #error here, creating new notebook to streamline this workflow

# %%
sdata = sd.SpatialData(
  images={"hne":img_for_sdata},
  points={"nuclei":points_for_sdata},
  tables={"adata":adata_cell_annotated}
)

sdata

# %%
sdata.pl.render_images("hne").pl.render_points("nuclei").pl.show()

# %%
#TODO: examine genes of interest. IDK the genes of interest, ask Yashika

# %%
sq.pl.spatial_scatter(sdata['adata'], shape=None, color='leiden')

# %%
sdata.pl.render_images("hne").pl.render_points(color="predicted_labels").pl.show()

# %%
sq.pl.spatial_scatter(sdata['adata'], shape=None, color='predicted_labels')

# %%
sq.gr.spatial_neighbors(sdata['adata'])
sq.gr.nhood_enrichment(sdata['adata'], cluster_key="predicted_labels")
sq.pl.nhood_enrichment(sdata['adata'], cluster_key="predicted_labels")

# %%
sq.gr.spatial_neighbors(sdata['adata'])
sq.gr.nhood_enrichment(sdata['adata'], cluster_key="leiden")
sq.pl.nhood_enrichment(sdata['adata'], cluster_key="leiden")

# %%
sq.gr.co_occurrence(sdata['adata'], cluster_key="predicted_labels")
sq.pl.co_occurrence(
    sdata['adata'],
    cluster_key="predicted_labels",
    clusters="Hepatocytes",
    figsize=(8, 4),
)

# %%


# %%
# sq.gr.ligrec(
#     sdata['adata'],
#     n_perms=100,
#     cluster_key="predicted_labels",
#     use_raw=False
# )
# sq.pl.ligrec(
#     sdata['adata'],
#     cluster_key="predicted_labels",
#     source_groups="Hepatocytes",
#     remove_empty_interactions=True,
#     remove_nonsig_interactions=True,
#     swap_axes=True
# )

# %%

