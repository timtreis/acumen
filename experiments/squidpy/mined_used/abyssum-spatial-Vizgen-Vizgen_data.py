# mined from: https://github.com/abyssum/spatial/blob/8d0f5e8eeefeda47a2960cd4a6cabda863a18c13/Vizgen/Vizgen_data.ipynb
# symbols: squidpy.gr.centrality_scores, squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.ripley, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.centrality_scores, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.ripley, squidpy.pl.spatial_scatter, squidpy.read.vizgen

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import squidpy as sq

sc.logging.print_header()

# %%
# # # Download and unpack the Vizgen data
# !mkdir tutorial_data
# !mkdir tutorial_data/vizgen_data
# !mkdir tutorial_data/vizgen_data/images

# %%
vizgen_dir = Path().resolve() / "tutorial_data" / "vizgen_data"

adata = sq.read.vizgen(
    path=vizgen_dir,
    counts_file="datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_by_gene_S1R1.csv",
    meta_file="datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_metadata_S1R1.csv",
    transformation_file="datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_micron_to_mosaic_pixel_transform.csv",
)

# %%
# Calculate quality control metrics

# %%
sc.pp.calculate_qc_metrics(adata, percent_top=(50, 100, 200, 300), inplace=True)

# %%
adata.obsm["blank_genes"].to_numpy().sum() / adata.var["total_counts"].sum() * 100

# %%
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].set_title("Total transcripts per cell")
sns.histplot(
    adata.obs["total_counts"],
    kde=False,
    ax=axs[0],
)

axs[1].set_title("Unique transcripts per cell")
sns.histplot(
    adata.obs["n_genes_by_counts"],
    kde=False,
    ax=axs[1],
)

axs[2].set_title("Transcripts per FOV")
sns.histplot(
    adata.obs.groupby("fov").sum()["total_counts"],
    kde=False,
    ax=axs[2],
)

axs[3].set_title("Volume of segmented cells")
sns.histplot(
    adata.obs["volume"],
    kde=False,
    ax=axs[3],
)

# %%
sc.pp.filter_cells(adata, min_counts=10)

# %%
# !pip install scikit-misc

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# %%
# Visualize annotation on UMAP and spatial coordinates

# %%
sc.pl.umap(
    adata,
    color=[
        "total_counts",
        "n_genes_by_counts",
        "leiden",
    ],
    wspace=0.4,
)

# %%
sq.pl.spatial_scatter(
    adata,
    shape=None,
    color=[
        "leiden",
    ],
    wspace=0.4,
)

# %%
# Computation of spatial statistics

# %%
# Building the spatial neighbors graphs

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

# %%
# Compute centrality scores

# %%
sq.gr.centrality_scores(adata, cluster_key="leiden")

# %%
sq.pl.centrality_scores(adata, cluster_key="leiden", figsize=(16, 5))

# %%
# Compute co-occurrence probability

# %%
adata_subsample = sc.pp.subsample(adata, fraction=0.5, copy=True)

# %%
sq.gr.co_occurrence(
    adata_subsample,
    cluster_key="leiden",
)
sq.pl.co_occurrence(
    adata_subsample,
    cluster_key="leiden",
    clusters="12",
    figsize=(10, 10),
)
sq.pl.spatial_scatter(
    adata_subsample,
    color="leiden",
    shape=None,
    size=2,
)

# %%
# Neighbors enrichment analysis

# %%
sq.gr.nhood_enrichment(adata, cluster_key="leiden")

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 7))
sq.pl.nhood_enrichment(
    adata,
    cluster_key="leiden",
    figsize=(8, 8),
    title="Neighborhood enrichment adata",
    ax=ax[0],
)
sq.pl.spatial_scatter(adata_subsample, color="leiden", shape=None, size=2, ax=ax[1])

# %%
# Compute Ripley’s statistics

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
mode = "L"

sq.gr.ripley(adata, cluster_key="leiden", mode=mode)
sq.pl.ripley(adata, cluster_key="leiden", mode=mode, ax=ax[0])

sq.pl.spatial_scatter(
    adata_subsample,
    color="leiden",
    groups=["0", "1", "3"],
    shape=None,
    size=2,
    ax=ax[1],
)

# %%
# Compute Moran’s I score

# %%
sq.gr.spatial_neighbors(adata_subsample, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(
    adata_subsample,
    mode="moran",
    n_perms=100,
    n_jobs=1,
)
adata_subsample.uns["moranI"].head(10)

# %%
sq.pl.spatial_scatter(
    adata_subsample,
    color=[
        "Slc17a7",
        "Npy2r",
    ],
    shape=None,
    size=2,
    img=False,
)

# %%


# %%

