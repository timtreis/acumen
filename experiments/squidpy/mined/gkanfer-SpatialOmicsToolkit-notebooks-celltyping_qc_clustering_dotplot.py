# mined from: https://github.com/gkanfer/SpatialOmicsToolkit/blob/a41fd656300d917c1b94af94afead5b9da63eec5/notebooks/celltyping_qc_clustering_dotplot.ipynb
# symbols: squidpy.gr.spatial_neighbors, squidpy.pl.spatial_scatter

# %%
from skimage import io
import numpy as np
import os
import scanpy as sc
import squidpy as sq
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import gzip
import numpy as np
import celltypist
from celltypist import models


plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# %%
def calcQCmat(andata):
    andata.var_names_make_unique()
    andata.var["mt"] = andata.var_names.str.startswith("mt-")
    andata.var["ribo"] = andata.var_names.str.startswith(("RPS", "RPL"))
    andata.var["hb"] = andata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(andata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    return andata
path_016 = "/data/kanferg/Sptial_Omics/playGround/Data/Visium_HD_Mouse_Brain_square_example/square_016um"
andata016_ = sc.read_visium(path=path_016)
andata016 = calcQCmat(andata016_)
print(f"{np.shape(andata016.X.todense())}")
sc.pp.filter_cells(andata016, min_counts = 50)
sc.pp.filter_cells(andata016, min_genes = 80)

# %%
plt.rcParams['font.size'] = 12
fig, axs = plt.subplots(1, 2, figsize=(7, 3))  # Adjusted figsize for better readability
axs[0].set_title('Number of Cells with Detected \n Gene Expression')
axs[1].set_title('Number of Cells with Detected \n Gene Expression')
sns.histplot(andata016.var['n_cells_by_counts'], kde=False, ax=axs[0],bins = 50)
sns.histplot(andata016.var['n_cells_by_counts'][andata016.var['n_cells_by_counts'] < 50], kde=False, ax=axs[1])
plt.subplots_adjust(wspace=0.5)
plt.suptitle("Number of Cells with Detected Gene Expression", y=1.10)

# %%
andata016 = andata016[:,andata016.var.n_cells_by_counts > 50]
print(f'{andata016}')

# %%
print(f"{np.shape(andata016.X.todense())}")
andata016 = andata016[andata016.obs["pct_counts_mt"] < 20]
print(f"{np.shape(andata016.X.todense())}")

# %%
sc.pp.normalize_total(andata016)
sc.pp.log1p(andata016)
log1p_data = andata016.X.todense()
sc.pp.highly_variable_genes(andata016)
sc.pp.scale(andata016)
andata016.obsm['spatial'] = np.array(andata016.obsm['spatial'], dtype=np.float64)
sc.pp.pca(andata016, n_comps=20)
sc.pp.neighbors(andata016)
sc.tl.umap(andata016)
sc.tl.leiden(andata016, key_added="clusters", flavor="igraph", directed=False, n_iterations=2)

# %%
from matplotlib.colors import ListedColormap

# Combine multiple palettes to create a larger custom palette
palette = sns.color_palette("tab20") + sns.color_palette("tab20b") + sns.color_palette("tab20c")

# Convert the combined palette to a ListedColormap
listed_cmap = ListedColormap(palette)

# Create the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

# Plot the spatial scatter plot on the specified axis
sq.pl.spatial_scatter(andata016, color="clusters", ax=ax, palette=listed_cmap)

# %%
sc.tl.rank_genes_groups(
    andata016, groupby="clusters", method="wilcoxon", key_added="dea_clusters"
)

# %%
sc.pl.rank_genes_groups_dotplot(
    andata016, groupby="clusters", standard_scale="var", n_genes=5, key="dea_clusters"
)

# %%
sc.tl.filter_rank_genes_groups(
    andata016,
    min_in_group_fraction=0.01,
    max_out_group_fraction=0.01,
    key="dea_clusters",
    key_added="dea_clusters_filtered",
)

# %%
andata016.uns['dea_clusters']['logfoldchanges']

# %%
# Visualize the filtered genes:
sc.pl.rank_genes_groups_dotplot(
    andata016,
    groupby="clusters",
    standard_scale="var",
    n_genes=5,
    key="dea_clusters_filtered",
)

# %%
from matplotlib.colors import ListedColormap

# Combine multiple palettes to create a larger custom palette
palette = sns.color_palette("tab20") + sns.color_palette("tab20b") + sns.color_palette("tab20c")

# Convert the combined palette to a ListedColormap
listed_cmap = ListedColormap(palette)

# %%
sns.set_context("paper", font_scale=1)

resolutions = [0.1, 0.5, 1.0, 2.0]

res = resolutions[0]

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

sc.tl.leiden(andata016, key_added=f'clusters_res_{res}', flavor="igraph", directed=False, resolution=res, n_iterations=2)
sq.pl.spatial_scatter(andata016, color=f'clusters_res_{res}', ax=ax, palette=listed_cmap)
ax.set_title(f'Leiden Clustering (resolution={res})')

# %%
sns.set_context("paper", font_scale=1)

resolutions = [0.1, 0.5, 1.0, 2.0]

res = resolutions[1]

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

sc.tl.leiden(andata016, key_added=f'clusters_res_{res}', flavor="igraph", directed=False, resolution=res, n_iterations=2)
sq.pl.spatial_scatter(andata016, color=f'clusters_res_{res}', ax=ax, palette=listed_cmap)
ax.set_title(f'Leiden Clustering (resolution={res})')

# %%
sns.set_context("paper", font_scale=1)

resolutions = [0.1, 0.5, 1.0, 2.0]

res = resolutions[2]

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

sc.tl.leiden(andata016, key_added=f'clusters_res_{res}', flavor="igraph", directed=False, resolution=res, n_iterations=2)
sq.pl.spatial_scatter(andata016, color=f'clusters_res_{res}', ax=ax, palette=listed_cmap)
ax.set_title(f'Leiden Clustering (resolution={res})')

# %%
sns.set_context("paper", font_scale=1)

resolutions = [0.1, 0.5, 1.0, 2.0]

res = resolutions[1]

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

sc.tl.leiden(andata016, key_added=f'clusters_res_{res}', flavor="igraph", directed=False, resolution=res, n_iterations=2)
sq.pl.spatial_scatter(andata016, color=f'clusters_res_{res}', ax=ax, palette=listed_cmap)
ax.set_title(f'Leiden Clustering (resolution={res})')

# %%
sc.pl.rank_genes_groups_dotplot(
    andata016, groupby="clusters_res_0.5", standard_scale="var", n_genes=2, key="dea_clusters"
)

# %%
andata016.obsm['spatial']

# %%
sq.gr.spatial_neighbors(andata016,coord_type="grid", key_added='spatial_neighbors')

# %%
andata016.obs['clusters_spatial']

# %%
sc.tl.leiden(andata016, key_added=f'clusters_spatial', neighbors_key=andata016.obsp['spatial_neighbors_connectivities'])

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
sq.pl.spatial_scatter(andata016, color=f'clusters_spatial', ax=ax, palette=listed_cmap)
ax.set_title(f'')

# %%
andata016.obsm['spatial'] = np.array(andata016.obsm['spatial'], dtype=np.float64)

sq.gr.spatial_neighbors(andata016,coord_type="grid", n_neighs=6, n_rings=8, key_added='spatial_neighbors')
# Perform clustering using the spatial neighbors graph
sc.tl.leiden(andata016, key_added=f'clusters_spatial', adjacency=andata016.obsp['spatial_neighbors_connectivities'])

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
sq.pl.spatial_scatter(andata016, color=f'clusters_spatial', ax=ax, palette=listed_cmap)
ax.set_title(f'')

# %%
def calcQCmat(andata):
    andata.var_names_make_unique()
    andata.var["mt"] = andata.var_names.str.startswith("mt-")
    andata.var["ribo"] = andata.var_names.str.startswith(("RPS", "RPL"))
    andata.var["hb"] = andata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(andata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    return andata
path_016 = "/data/kanferg/Sptial_Omics/playGround/Data/Visium_HD_Mouse_Brain_square_example/square_016um"
andata016_ = sc.read_visium(path=path_016)
andata016 = calcQCmat(andata016_)
print(f"{np.shape(andata016.X.todense())}")
sc.pp.filter_cells(andata016, min_counts = 50)
sc.pp.filter_cells(andata016, min_genes = 80)
