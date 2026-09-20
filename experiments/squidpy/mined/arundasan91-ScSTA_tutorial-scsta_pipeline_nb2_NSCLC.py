# mined from: https://github.com/arundasan91/ScSTA_tutorial/blob/617a56436e9e09115aa389c656e7457b29c2b7a4/scsta_pipeline_nb2_NSCLC.ipynb
# symbols: squidpy.gr.centrality_scores, squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.ripley, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.centrality_scores, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.ripley, squidpy.pl.spatial_scatter, squidpy.pl.spatial_segment, squidpy.read.nanostring

# %%
# %matplotlib inline
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import scanpy as sc
import numpy as np
import squidpy as sq
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import anndata
from sklearn.cluster import KMeans
import monkeybread as mb 
import gseapy as gp
from gseapy.plot import barplot

import tqdm as notebook_tqdm

pd.set_option("display.max_columns", None)

# Plotting options, change to your liking
sc.settings.set_figure_params(dpi=80, frameon=False, facecolor="white")
sc.set_figure_params(dpi=80)
sc.set_figure_params(figsize=(4, 4))
sc.settings.verbosity = 0

# %%
nanostring_dir = Path().resolve() / "tutorial_data" / "nanostring_data"
sample_dir = nanostring_dir / "Lung9_Rep1" / "Lung9_Rep1-Flat_files_and_images"

adata = sq.read.nanostring(
    path=sample_dir,
    counts_file="Lung9_Rep1_exprMat_file.csv",
    meta_file="Lung9_Rep1_metadata_file.csv",
    fov_file="Lung9_Rep1_fov_positions_file.csv",
)

# %%
adata

# %%
adata.obs

# %%
adata.obs['tissue'] = "Lung9_Rep1"

# %%
adata.obs.head()

# %%
df = pd.read_csv(f'{sample_dir}/Lung9_Rep1_ctypes.csv', index_col=0)
df.index = [f'{str(c)}_{str(fov)}' for c,fov in zip(df['cell_ID'], df['fov'])]
df

# %%
adata_ctype = pd.merge(adata.obs, df['cell_type'], how='outer', right_index=True, left_index=True)
adata_ctype = adata_ctype.drop_duplicates()
adata_ctype['cell_type'].fillna('unknown', inplace=True)
adata_ctype.replace({k:'tumor' for k in ['tumor '+ str(i) for i in [5, 6, 9, 12, 13]]}, inplace=True)
adata_ctype = adata_ctype.loc[adata.obs.index]

for column_name in adata_ctype.columns:
    if column_name in adata.obs.columns:
        new_dtype = adata.obs[column_name].dtype
        adata_ctype[column_name] = adata.obs[column_name].astype(new_dtype)
        
adata.obs = adata_ctype

# %%
adata.obs

# %%
adata

# %%
adata.obs['cell_type'].unique()

# %%
adata.var["NegPrb"] = adata.var_names.str.startswith("NegPrb")
sc.pp.calculate_qc_metrics(adata, qc_vars=["NegPrb"], inplace=True)

# %%
adata = adata[:, ~adata.var_names.str.startswith("NegPrb")]

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

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
    adata.obs.groupby("fov")["total_counts"].sum(),
    kde=False,
    ax=axs[2],
)

plt.tight_layout()

# %%
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].set_title("Membrane Stain")
sns.histplot(
    adata.obs["Mean.MembraneStain"],
    kde=False,
    ax=axs[0],
)

axs[1].set_title("PanCK")
sns.histplot(
    adata.obs["Mean.PanCK"],
    kde=False,
    ax=axs[1],
)

axs[2].set_title("CD45")
sns.histplot(
    adata.obs["Mean.CD45"],
    kde=False,
    ax=axs[2],
)

axs[3].set_title("CD3")
sns.histplot(
    adata.obs["Mean.CD3"],
    kde=False,
    ax=axs[3],
)

plt.tight_layout()

# %%
sc.pp.filter_cells(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=100)

# %%
adata.obs

# %%
adata

# %%
selected_fovs = [f"{i}" for i in [17, 18, 19, 13, 14, 15]]

# %%
plot_focus = "cell_type"

axs_scatter = sq.pl.spatial_segment(
    adata,
    color=plot_focus,
    library_key="fov",
    library_id = selected_fovs,
    seg_cell_id="cell_ID", 
    seg_outline = True,
    # palette='tab20',
    img=False,
    colorbar=False,
    hspace=0, wspace=0,
    ncols=3,
    legend_loc=None,
    outline=False,
    axis_label = None,
    figsize=(7,4.5),
    # library_first=True,
    frameon=False,
    title=None,
    return_ax = True
)

# Get the current Matplotlib axes object
for ax in axs_scatter:    
    # Remove the title from the Matplotlib axes
    ax.set_title("")

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.0001, wspace=0.0001)

# Optionally, tighten the layout
# plt.tight_layout()

# plt.savefig(os.path.join(sample_dir, 'generated_figures', f'whole_tissue_{plot_focus}.png'), dpi=200)
# Show or save the plot
plt.show()

# %%
adata

# %%
adata = adata[adata.obs['fov'].isin(selected_fovs)]

# %%
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

# Create the scatter plot without grid lines
ax = sc.pl.scatter(
    adata,
    x='CenterX_global_px',
    y='CenterY_global_px',
    color='cell_type',
    show=False, 
    size=15,
    frameon=False, 
    ax=ax
    # legend_loc='none'
)

# Turn off x and y axes
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

os.makedirs(os.path.join(sample_dir, 'generated_figures'), exist_ok=True)
plt.savefig(os.path.join(sample_dir, 'generated_figures', f'scatter_plot_celltypes.png'), dpi=200)

# %%
plot_focus = "cell_type"

axs_scatter = sq.pl.spatial_segment(
    adata,
    color=plot_focus,
    library_key="fov",
    library_id = selected_fovs,
    seg_cell_id="cell_ID", 
    seg_outline = True,
    # palette='tab20',
    img=False,
    colorbar=False,
    hspace=0, wspace=0,
    ncols=3,
    legend_loc=None,
    outline=False,
    axis_label = None,
    figsize=(7,4.5),
    # library_first=True,
    frameon=False,
    title=None,
    return_ax = True
)

# Get the current Matplotlib axes object
for ax in axs_scatter:    
    # Remove the title from the Matplotlib axes
    ax.set_title("")

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.0001, wspace=0.0001)

# Optionally, tighten the layout
# plt.tight_layout()

# plt.savefig(os.path.join(sample_dir, 'generated_figures', f'whole_tissue_{plot_focus}.png'), dpi=200)
# Show or save the plot
plt.show()

# %%
adata.layers["counts"] = adata.X.copy()

# %%
sc.pp.normalize_total(adata)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.pl.highly_variable_genes(adata)

# %%
sc.pp.pca(adata, svd_solver='arpack')

# %%
sc.pl.pca(adata, color='NDRG1')

# %%
sc.pl.pca(adata, color='COL1A1')

# %%
# %%time
sc.pp.neighbors(adata, n_neighbors=100)

# %%
adata

# %%
# %%time
sc.tl.umap(adata)

# %%
# %%time
sc.tl.louvain(adata)

# %%
# %%time
sc.tl.leiden(adata)

# %%
sc.pl.umap(
    adata,
    color=[
        "total_counts",
        "n_genes_by_counts",
        "leiden",
        "louvain",
    ],
    wspace=0.4,
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
dotsize = 10

sc.pl.umap(
    adata,
    color=[
        'cell_type'
    ], size=dotsize,
    wspace=0.4, ax=ax, palette='tab20'
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
dotsize = 10
ax = sc.pl.umap(adata, color=['cell_type'], groups=["T CD4 memory", "T CD4 naive", "T CD8 memory", "T CD8 naive"], show=False, size=dotsize, ax=ax)

# We can change the 'NA' in the legend that represents all cells outside of the
# specified groups
legend_texts=ax.get_legend().get_texts()
# Find legend object whose text is "NA" and change it
for legend_text in legend_texts:
    if legend_text.get_text()=="NA":
        legend_text.set_text('other cell types')

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
dot_size=10
# Plot all cells as background
ax=sc.pl.umap(adata, show=False,s=dot_size, ax=ax)

# Plot ontop expression of a single cell group by subsetting adata
sc.pl.umap(adata[adata.obs.cell_type=='tumor',:],color='NDRG1', ax=ax, s=dot_size)

# %%
# %%time
sq.pl.spatial_segment(
    adata,
    color='leiden',
    library_key="fov",
    library_id = selected_fovs,
    seg_cell_id="cell_ID", 
    seg_outline = True,
    # palette='tab20',
    img=False,
    colorbar=False,
    hspace=0, wspace=0,
    ncols=3,
    legend_loc=None,
    outline=False,
    axis_label = None,
    figsize=(7,4.5),
    # library_first=True,
    frameon=False,
    title='',
)

# %%
# %%time
sq.pl.spatial_segment(
    adata,
    color='louvain',
    library_key="fov",
    library_id = selected_fovs,
    seg_cell_id="cell_ID", 
    seg_outline = True,
    # palette='tab20',
    img=False,
    colorbar=False,
    hspace=0.0001, wspace=0.0001,
    ncols=3,
    legend_loc=None,
    outline=False,
    axis_label = None,
    figsize=(7,4.5),
    # library_first=True,
    frameon=False,
    title='',
)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 7))

for _ax in ax:
    _ax.set_facecolor('white')

selected_fov = 19
    # [17, 18, 19, 13, 14, 15]
    
sq.pl.spatial_segment(
    adata,
    shape="hex",
    color="leiden",
    library_key="fov",
    library_id=f"{selected_fov}",
    seg_cell_id="cell_ID",
    img=False,
    size=60,
    ax=ax[0],
)

sq.pl.spatial_segment(
    adata,
    shape="hex",
    color="louvain",
    library_key="fov",
    library_id=f"{selected_fov}",
    seg_cell_id="cell_ID",
    img=False,
    size=60,
    ax=ax[1],
)

sq.pl.spatial_segment(
    adata,
    shape="hex",
    color="cell_type",
    library_key="fov",
    library_id=f"{selected_fov}",
    seg_cell_id="cell_ID",
    img=False,
    size=60,
    ax=ax[2],
)

plt.tight_layout()

# %%
# %%time
sq.pl.spatial_segment(
    adata,
    color='Max.PanCK',
    library_key="fov",
    library_id = selected_fovs,
    seg_cell_id="cell_ID", 
    seg_outline = True,
    img=False,
    colorbar=False,
    hspace=0, wspace=0,
    ncols=3,
    legend_loc=None,
    outline=False,
    axis_label = None,
    figsize=(7,4.5),
    frameon=False,
    title='',
)

# %%
adata

# %%
# %%time
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
sq.gr.spatial_neighbors(
    adata,
    n_neighs=15,
    coord_type="generic",
)
_, idx = adata.obsp["spatial_connectivities"][1111, :].nonzero()
idx = np.append(idx, 1111)
sq.pl.spatial_scatter(
    adata[idx, :],
    library_id="16",
    color="cell_type",
    connectivity_key="spatial_connectivities",
    size=3,
    edges_width=1,
    edges_color="black",
    img=False,
    title="K-nearest neighbors",
    ax=ax[0],
)

sq.gr.spatial_neighbors(
    adata,
    n_neighs=15,
    coord_type="generic",
    delaunay=True,
)
_, idx = adata.obsp["spatial_connectivities"][1111, :].nonzero()
idx = np.append(idx, 1111)
sq.pl.spatial_scatter(
    adata[idx, :],
    library_id="16",
    color="cell_type",
    connectivity_key="spatial_connectivities",
    size=3,
    edges_width=1,
    edges_color="black",
    img=False,
    title="Delaunay triangulation",
    ax=ax[1],
)

plt.tight_layout()

# %%
# %%time
sq.gr.centrality_scores(adata, cluster_key="cell_type")

# %%
# %%time
sq.pl.centrality_scores(adata, cluster_key="cell_type", figsize=(15, 6))

# %%
adataset = adata[adata.obs.fov.isin(["17", "18"])].copy()

# %%
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

# Create the scatter plot without grid lines
ax = sc.pl.scatter(
    adataset,
    x='CenterX_global_px',
    y='CenterY_global_px',
    color='cell_type',
    show=False, 
    size=15,
    frameon=False, 
    ax=ax
    # legend_loc='none'
)

# Turn off x and y axes
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

# %%
sq.gr.co_occurrence(
    adataset,
    cluster_key="cell_type",
)

# %%
sq.pl.co_occurrence(
    adataset,
    cluster_key="cell_type",
    clusters='tumor', figsize=(15, 7), 
)

# %%
# %%time
sq.gr.nhood_enrichment(adata, cluster_key="cell_type")

# %%
# %%time
sq.gr.nhood_enrichment(adataset, cluster_key="cell_type")

# %%
fig, ax = plt.subplots(1, 2, figsize=(22, 22))
sq.pl.nhood_enrichment(
    adata,
    cluster_key="cell_type",
    figsize=(3, 3), vmin=12, vmax=-12, vcenter=0,
    ax=ax[0],
    title="Neighborhood enrichment adata", cmap='RdBu_r'
)
sq.pl.nhood_enrichment(
    adataset,
    cluster_key="cell_type",
    figsize=(3, 3), vmin=12, vmax=-12, vcenter=0,
    ax=ax[1],
    title="Neighborhood enrichment adataset", cmap='RdBu_r'
)

plt.tight_layout()

# %%
# %%time
mode = "L"
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

sq.gr.ripley(adataset, cluster_key="cell_type", mode=mode)
sq.pl.ripley(
    adataset,
    cluster_key="cell_type",
    mode=mode,
    ax=ax[0],
)

sq.pl.spatial_segment(
    adataset,
    shape="hex",
    color="cell_type",
    library_id=["17"],
    library_key="fov",
    seg_cell_id="cell_ID",
    img=False,
    size=60,
    ax=ax[1],
)

plt.tight_layout()

# %%
# %%time
sq.gr.spatial_neighbors(adataset, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(
    adataset,
    mode="moran",
    n_perms=100,
    n_jobs=1,
)
adataset.uns["moranI"].head(10)

# %%
# %%time
sq.gr.spatial_neighbors(adataset, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(
    adataset,
    mode="moran",
    n_perms=100,
    n_jobs=1,
)
adataset.uns["moranI"].head(10)

# %%
adataset.uns["moranI"].index[:10]

# %%
# %%time
sq.pl.spatial_segment(
    adataset,
    shape="hex",
     color=["NDRG1", "HSP90AB1", "CXCL10"],
    library_id=["17"],
    library_key="fov",
    seg_cell_id="cell_ID", 
    palette=None,
    legend_loc=None,
    img=False,
    ncols=4,
    size=40,)

plt.tight_layout()

# %%

def calculate_neighborhood_cell_composition_anndata(adata, n_neighbors=200, spatial_key='spatial_fov', cell_type_col='cell_type'):
    """
    This function calculates the composition of cell types in the neighborhoods 
    of each cell in a given AnnData object and returns the modified AnnData.
    
    Parameters:
    - adata: AnnData object containing cell data, including x and y coordinates and cell types.
    - n_neighbors: Number of nearest neighbors to consider for each cell. Default is 200.
    
    Returns:
    - adata: AnnData object with added columns representing the composition of cell types 
             in the neighborhoods of each cell.
    """

    # Extracting coordinates and cell types from the AnnData object
    coords = adata.obsm[spatial_key]
    cell_types = adata.obs[cell_type_col].values
    
    # Obtaining unique cell types and sorting them
    unique_cell_types = sorted(adata.obs[cell_type_col].unique())

    # Initializing a NearestNeighbors object and fitting it to the data
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(coords)

    # Finding the indices of nearest neighbors for each point
    _, neighbors_indices = neigh.kneighbors(coords)

    # Mapping cell types to indices for faster processing
    cell_type_to_index = {cell_type: i for i, cell_type in enumerate(unique_cell_types)}
    cell_type_indices = np.vectorize(cell_type_to_index.get)(cell_types)

    # Initializing an array to hold the counts of cell types in neighborhoods
    cell_composition_counts = np.zeros((len(adata), len(unique_cell_types)))

    # Printing progress information
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating Neighborhood Cell Composition")

    # Counting the occurrences of each cell type in the neighborhoods
    for i, neighbors in enumerate(neighbors_indices):
        # Updating progress every 30000 iterations
        if i % 3000 == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing cell ID: {i}")
        neighbor_types = cell_type_indices[neighbors]
        for neighbor_type in neighbor_types:
            cell_composition_counts[i, neighbor_type] += 1

    # Creating a DataFrame from the counts
    cell_composition_df = pd.DataFrame(
        cell_composition_counts, 
        columns=[f'n_{ct}' for ct in unique_cell_types], 
        index=adata.obs.index
    )

    # Adding the new DataFrame to the AnnData object's .obs attribute
    for col in cell_composition_df.columns:
        adata.obs[col] = cell_composition_df[col].values

    # Printing completion message
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed Neighborhood Cell Composition")

    return adata

# %%
adata

# %%
# %%time
adata = calculate_neighborhood_cell_composition_anndata(adata, n_neighbors=100, spatial_key='spatial_fov', cell_type_col='cell_type')

# %%
adata

# %%
niche_columns = ['n_'+ctype for ctype in adata.obs.cell_type.unique()]

adata.obs[niche_columns]

# %%
import matplotlib.pyplot as plt
import numpy as np
import math

unique_ctypes = list(adata.obs.cell_type.unique())

# Set the number of columns
ncols = 3
# Calculate the number of rows based on the number of unique cell types and columns
nrows = math.ceil(len(unique_ctypes) / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3 * nrows))

# Coordinates of all cells
x_all = adata.obs['CenterX_global_px']
y_all = adata.obs['CenterY_global_px']

# Flatten the axes array for easy iteration (in case of multiple rows and columns)
axes = axes.flatten()

# Plot each niche in its own subplot
for i, ctype in enumerate(unique_ctypes):
    ax = axes[i]
    
    # Plot all cells in gray
    ax.scatter(
        x_all,
        y_all,
        c='lightgray',
        s=2,
        label='Other cells'
    )
    
    # Get the data for the current cell type
    ctype_data = adata.obs[adata.obs['cell_type'] == ctype]
    x = ctype_data['CenterX_global_px']
    y = ctype_data['CenterY_global_px']
    values = ctype_data[f'n_{ctype}']  # Values to use for the colormap
    
    # Plot the current cell type with a colormap
    scatter = ax.scatter(
        x,
        y,
        c=values,  # Use the values from the column for color mapping
        s=2,
        vmin=0,
        vmax=100,
        cmap='rainbow',  # Specify the colormap
        label=f'n_{ctype}'
    )
    
    # Add a colorbar to show the range of the values
    plt.colorbar(scatter, ax=ax, label=f'n_{ctype}')
    
    # Turn off x and y axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'n_{ctype}')  # Add the niche label as the subplot title

# Turn off unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# %%


# %%


# %%
# %%time
n_clusters = 5
random_state = 1111

kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
adata.obs['niche_kmeans'] = kmeans.fit_predict(adata.obs[niche_columns])

# Prepend 'niche_' to the K-means cluster labels
adata.obs['niche_kmeans'] = adata.obs['niche_kmeans'].astype(str).astype('category')
adata.obs['niche_kmeans'] = adata.obs['niche_kmeans'].cat.rename_categories(lambda x: f'niche_{x}')

# %%
def plot_cell_type_proportions(adata, sort_by_cell_type=None):
    # Group by 'niche_kmeans' and 'cell_type' to count occurrences
    count_df = adata.obs.groupby(['niche_kmeans', 'cell_type']).size().reset_index(name='counts')
    
    # Calculate proportions for each 'niche_kmeans'
    total_counts = count_df.groupby('niche_kmeans')['counts'].transform('sum')
    count_df['proportion'] = count_df['counts'] / total_counts
    
    # Create a pivot table for plotting
    pivot_df = count_df.pivot(index='niche_kmeans', columns='cell_type', values='proportion').fillna(0)
    
    # Sort pivot_df based on the specified cell type if provided
    if sort_by_cell_type and sort_by_cell_type in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by=sort_by_cell_type, ascending=False)
    
    # Plotting the proportion bar plot
    pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.ylabel('Proportion')
    plt.xlabel('Niche KMeans Cluster')
    plt.title('Proportion of Cell Types in Each Niche Cluster')
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Example usage: Sort by a specific cell type
plot_cell_type_proportions(adata, sort_by_cell_type='tumor')

# %%
_ = adata.obs.niche_kmeans.value_counts().plot(kind='bar', title='Number of cells in each niche')

# %%
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

# Create the scatter plot without grid lines
ax = sc.pl.scatter(
    adata,
    x='CenterX_global_px',
    y='CenterY_global_px',
    color='niche_kmeans',
    show=False, 
    size=15,
    frameon=False, 
    ax=ax,
)

# Turn off x and y axes
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

# %%
niche_labels = ['niche_1', 'niche_2', 'niche_4', 'niche_0', 'niche_3']

fig, axes = plt.subplots(len(niche_labels), 1, figsize=(7, 3 * len(niche_labels)))

# If there's only one niche, axes will not be an array; we convert it to a list for uniform handling
if len(niche_labels) == 1:
    axes = [axes]

# Plot each niche in its own subplot
for i, niche in enumerate(niche_labels):
    ax = axes[i]
    sc.pl.scatter(
        adata,
        x='CenterX_global_px',
        y='CenterY_global_px',
        color='niche_kmeans',
        groups=[niche],
        show=False,
        size=15,
        frameon=False,
        ax=ax,
        legend_loc='none'  # Hide legend to avoid repetition in each subplot
    )
    # Turn off x and y axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(niche)  # Add the niche label as the subplot title

plt.tight_layout()
plt.show()

# %%
# %%time
groupby='niche_kmeans'
method='wilcoxon'
n_genes = 5

# Perform differential expression analysis
sc.tl.dendrogram(adata, groupby=groupby)
sc.tl.rank_genes_groups(adata, 
                        groupby=groupby, 
                        method=method, 
                        n_genes=adata.n_vars, 
                        tie_correct=True                        
                       )

# %%
sc.pl.rank_genes_groups_heatmap(
    adata,
    groupby=groupby,
    n_genes=n_genes,
    use_raw=False,
    show=True,
    dendrogram=True
)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata,
    groupby=groupby,
    n_genes=n_genes,
    use_raw=False,
    show=True
)

# %%
adata

# %%
adata_tumor = adata[adata.obs.cell_type=='tumor']

# %%
adata_tumor = adata_tumor[adata_tumor.obs['niche_kmeans'].isin(['niche_1', 'niche_3'])]
adata_tumor

# %%
# %%time
groupby = 'niche_kmeans'
sc.tl.dendrogram(adata_tumor, groupby=groupby)
sc.tl.rank_genes_groups(adata_tumor, groupby=groupby)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_tumor,
    groupby=groupby,
    values_to_plot='logfoldchanges',
    cmap='bwr',
    n_genes=n_genes,
    use_raw=False,
    vmin=-5, vmax=5,
    show=True
)

# %%


# %%
# %%time


# Step 1: Extract the top upregulated genes for each group
top_genes = {}
groupby = "cell_type"  # You can modify this to match your analysis
n_genes = 50           # Adjust as needed for your top n genes

# Get the ranked genes data
ranked_genes = adata_tumor.uns['rank_genes_groups']
groups = ranked_genes['names'].dtype.names

# Extract the top upregulated genes for each group
for group in groups:
    gene_names = ranked_genes['names'][group][:n_genes]
    logfold_changes = ranked_genes['logfoldchanges'][group][:n_genes]
    
    # Only keep upregulated genes
    upregulated_genes = [gene for gene, lfc in zip(gene_names, logfold_changes) if lfc > 0]
    top_genes[group] = upregulated_genes

# Step 2: Perform GSEA for each group
# Here, we use gene ontology (GO_Biological_Process_2021) as an example gene set library
gsea_results = {}
for group, genes in top_genes.items():
    if genes:
        enrichment_result = gp.enrichr(
            gene_list=genes,
            gene_sets=['GO_Biological_Process_2021', 'MSigDB_Hallmark_2020'],
            organism='Human',  # Change as needed: 'Mouse', etc.
            outdir=None,       # Disable file output
            verbose=False
        )
        gsea_results[group] = enrichment_result

        # Step 3: Plot the top enriched terms for each group
        if enrichment_result.res2d is not None and not enrichment_result.res2d.empty:
            print(f"Top Enriched Terms for {group}")
            barplot(enrichment_result.res2d, title=f"GSEA Enrichment - {group}")
        # print(enrichment_result.res2d)
plt.show()

# %%
'VEGFB' in adata.var_names

# %%
lrs = mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='IL2')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='IL2RA')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='IL6')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='IL6R')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='JAG1')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='NOTCH1')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='NOTCH2')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='NOTCH3')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='VEGFA')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='VEGFB')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='FLT1')
lrs += mb.util.load_ligand_receptor_pairs_omnipath(adata, require_gene='KDR')
lrs

# %%
lr_pairs = []
for g1, g2 in lrs:
    if g2 in ['IL2', 'IL6', 'JAG1']:
        lr_pairs.append((g2, g1))
    else:
        lr_pairs.append((g1, g2))
lr_pairs

# %%
# For each pDC, find the neighboring Tregs
cell_to_neighbors = mb.calc.cell_neighbors(
    adata,
    groupby='cell_type',
    group1=['tumor'],
    group2=['fibroblast', 'neutrophil', 'T CD4 memory',],
    radius=50, 
    basis='spatial_fov',
) 

lr_pair_to_score = mb.calc.ligand_receptor_score(
    adata,
    cell_to_neighbors,
    lr_pairs=lr_pairs
)

# %%
res = mb.stat.ligand_receptor_score(
    adata,
    cell_to_neighbors,
    actual_scores=lr_pair_to_score
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(13,13))
mb.plot.ligand_receptor_scatter(
    lr_pair_to_score,
    res,
    ax=ax
)

# %%

