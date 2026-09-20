# mined from: https://github.com/elixir-europe-training/ELIXIR-SCO-spatial-omics/blob/e1f726c84b6e44e25f16b8b63bcf0ffcac788512/day_3/practical_4/workdir/p4_niche_domain.ipynb
# symbols: squidpy.gr.centrality_scores, squidpy.gr.interaction_matrix, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.centrality_scores, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
# Data analysis and ML imports
import pandas as pd
import matplotlib.pyplot as plt

# single-cell imports
import squidpy as sq
import scanpy as sc

from pathlib import Path
import os

import warnings
warnings.filterwarnings("ignore")

# %%
PATH = "/data/spatial_workshop/day3/practical_4"

# %%
# load adata
adata = sc.read_h5ad(Path(PATH, 'xenium_mouse_ad_annotated_rotated.h5ad'))
adata

# %%
# Creating a DataFrame from 'split', 'fov', and 'condition'
df = adata.obs[['condition', 'time', 'batch_key']]
value_counts = pd.DataFrame(df.values, columns=df.columns).value_counts()
print(value_counts)

# %%
# Add "Unknown" as a category
adata.obs["cell_types"] = adata.obs["cell_types"].cat.add_categories("Unknown")

# Fill NaN values with "Unknown"
adata.obs["cell_types"] = adata.obs["cell_types"].fillna("Unknown")

# %%
if 'cell_types_colors' in adata.uns:
    del adata.uns['cell_types_colors']
    
sq.gr.spatial_neighbors(adata, n_neighs=5, coord_type="generic", key_added = 'neighs_based_spatial')
sq.pl.spatial_scatter(
    adata,
    shape=None,
    library_key = 'sample',
    color=["cell_types"],
    connectivity_key="neighs_based_spatial_connectivities",
    title=adata.obs['sample'].cat.categories,
    ncols=3,
    size = 10
)

# %%
sq.gr.spatial_neighbors(adata, radius=0.2, coord_type="generic", key_added = 'radius_based_spatial')
sq.pl.spatial_scatter(
    adata,
    shape=None,
    library_key = 'sample', 
    color="cell_types",
    connectivity_key="radius_based_spatial_connectivities",
    title=adata.obs['sample'].cat.categories,
    ncols=3,
    size=10,
)

# %%
sq.gr.interaction_matrix(adata, cluster_key="cell_types", connectivity_key="neighs_based_spatial")
sq.pl.interaction_matrix(adata, cluster_key="cell_types", connectivity_key="neighs_based_spatial")

# %%
#TODO: Interaction matrix for radius based graph

# %%
sq.gr.centrality_scores(adata, cluster_key = "cell_types", connectivity_key = "neighs_based_spatial")
sq.pl.centrality_scores(adata, cluster_key = "cell_types")

# %%
sq.gr.nhood_enrichment(adata, cluster_key="cell_types", library_key = 'condition', connectivity_key = "radius_based_spatial")
sq.pl.nhood_enrichment(
    adata, cluster_key="cell_types", method="average", figsize=(5, 5)
) 

# %%
import scvi
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
import squidpy as sq
import numpy as np
import cellcharter as cc
import os
import logging
logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
logger.setLevel(logging.ERROR)

# %%
scvi.settings.seed = 12345
scvi.settings.num_threads = 2

# %%
sc.pp.filter_cells(adata, min_counts=3)

# %%
scvi.model.SCVI.setup_anndata(adata)

# %%
LOAD_MODEL = True

# %%
if LOAD_MODEL:
    model = scvi.model.SCVI.load(os.path.join(PATH, 'scvi_model'), adata=adata)
else:
    model = scvi.model.SCVI(
        adata,
        n_layers=1,
        n_latent=10,
        use_layer_norm="both",
        use_batch_norm="none",
    )
    model.train(early_stopping=True, enable_progress_bar=True, max_epochs=10)

# %%
plt.figure(figsize=(5, 5))
plt.plot(
    model.history[f"reconstruction_loss_train"],
    label="train",
    color="darkgreen",
    linewidth=1.25
)
plt.plot(
    model.history[f"reconstruction_loss_validation"],
    label="validation",
    color="firebrick",
    linewidth=1.25
    )
plt.legend()
plt.title("reconstruction_loss")  
plt.tight_layout()

# %%
adata.obsm['X_scVI'] = model.get_latent_representation(adata).astype(np.float32)

# %%
sq.gr.spatial_neighbors(adata, library_key='sample', coord_type='generic')

# %%
sq.pl.spatial_scatter(
    adata, 
    shape=None, 
    library_key='sample',
    library_id=adata.obs['sample'].cat.categories[0],
    color="sample", 
    size=1, 
    figsize=(10,10),
    connectivity_key="spatial_connectivities",
    ncols=1
)

# %%
sq.gr.spatial_neighbors(adata, library_key='sample', coord_type='generic', percentile=99)

# %%
sq.pl.spatial_scatter(
    adata, 
    shape=None, 
    library_key='sample',
    library_id=adata.obs['sample'].cat.categories[0],
    color="sample", 
    size=1, 
    figsize=(10,10),
    connectivity_key="spatial_connectivities",
    ncols=1
)

# %%
cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter_temp', sample_key='sample')

# %%


# %%
gmm = cc.tl.Cluster(n_clusters=18, random_state=12345)
gmm.fit(adata, use_rep='X_cellcharter_temp')
adata.obs['spatial_domain_temp'] = gmm.predict(adata, use_rep='X_cellcharter_temp')

# %%
if 'spatial_domain_temp_colors' in adata.uns:
    del adata.uns['spatial_domain_temp_colors']

sq.pl.spatial_scatter(
    adata, 
    shape=None, 
    library_key='sample', 
    color=["spatial_domain_temp", "cell_types"], 
    size=1,
    figsize=(10,10),
    title=np.repeat(adata.obs['sample'].cat.categories, 2),
    ncols=2
)

# %%
model = scvi.model.SCVI.load(os.path.join(PATH, 'scvi_model'), adata=adata)
adata.obsm['X_scVI'] = model.get_latent_representation(adata).astype(np.float32)
cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter', sample_key='sample')

autok = cc.tl.ClusterAutoK.load(Path(PATH, 'autok_l3'))
cc.pl.autok_stability(autok)

# If it takes too long also in this case, you can load the domain labels from the data folder
# adata.obs['spatial_domain_18'] = pd.read_csv(Path(PATH, 'spatial_domains_cellcharter.csv'), index_col=0)['spatial_domain_18']
# adata.obs['spatial_domain_18'] = adata.obs['spatial_domain_18'].astype('category')

# %%
adata.obs['spatial_domain_18'] = autok.predict(adata, use_rep='X_cellcharter', k=18)

# %%
if 'spatial_domain_18_colors' in adata.uns:
    del adata.uns['spatial_domain_18_colors']

sq.pl.spatial_scatter(
    adata, 
    shape=None, 
    library_key='sample', 
    color=["spatial_domain_18", "cell_types"], 
    size=1,
    figsize=(10,10),
    title=np.repeat(adata.obs['sample'].cat.categories, 2),
    ncols=2
)

# %%
cc.gr.enrichment(
    adata,
    group_key='spatial_domain_18',
    label_key='cell_types',
)
cc.pl.enrichment(
    adata,
    group_key='spatial_domain_18',
    label_key='cell_types',
    dot_scale=8
)

# %%


# %%
sq.pl.spatial_scatter(
    adata, 
    shape=None, 
    library_key='sample', 
    color=["spatial_domain_9", "spatial_domain_18", "spatial_domain_23"], 
    size=1,
    figsize=(10,10),
    title=np.repeat(adata.obs['sample'].cat.categories, 3),
    ncols=3
)

# %%
cc.gr.connected_components(adata, cluster_key='spatial_domain_18', min_cells=100)

# %%
# Hackfix: squidpy's spatial_scatter has some issues with categorical data with NaNs.
adata.obs['component_tmp'] = adata.obs['component'].astype('str')

# %%
if 'component_tmp_colors' in adata.uns:
    del adata.uns['component_tmp_colors']

sq.pl.spatial_scatter(
    adata[(adata.obs['sample'].isin(['TgCRND8_17_9', 'TgCRND8_2_5']))], 
    shape=None, 
    library_key='sample', 
    color=["component_tmp", "spatial_domain_18"], 
    size=1,
    figsize=(10,10),
    title=np.repeat(['TgCRND8_17_9', 'TgCRND8_2_5'], 2),
    ncols=2
)

# %%
cc.tl.boundaries(adata, alpha_start=10)

# %%
from cellcharter_utils import plot_boundaries, plot_shape_metrics

# %%
plot_boundaries(adata, sample='wildtype_5_7', show_cells=True, cells_radius=10)

# %%
cc.tl.curl(adata)
cc.tl.linearity(adata)

# %%
plot_shape_metrics(adata, cluster_key='spatial_domain_18', figsize=(6,3), cluster_id=8, metrics=['curl', 'linearity'])

# %%


# %%


# %%
adata = sc.read_h5ad(Path(PATH, 'xenium_mouse_ad_annotated_rotated.h5ad'))

# %%
adata_section = adata[(adata.obs['time'] == '5_7') & (adata.obs['condition'] == 'wildtype')]
adata_section

# %%
adata_section.obsm['spatial'][:,0]

# %%
## add x and y coordinate to .obs (needed for plot later)
adata_section.obs['x'] = adata_section.obsm['spatial'][:, 0]
adata_section.obs['y'] = adata_section.obsm['spatial'][:, 1]

# %%
from banksy_utils.load_data import load_adata, display_adata

from banksy_utils.filter_utils import normalize_total, filter_hvg, print_max_min

# Normalizes the AnnData object
adata_section = normalize_total(adata_section)

# %%
coord_keys = ('x', 'y', 'spatial')

# set parameters 
plot_graph_weights = True
k_geom = 15 # number of neighbors
max_m = 1 # azumithal transform up to kth order
nbr_weight_decay = "scaled_gaussian" # can also be "reciprocal", "uniform" or "ranked"

# %%
from banksy.main import median_dist_to_nearest_neighbour

# Find median distance to closest neighbours, the median distance will be `sigma`
nbrs = median_dist_to_nearest_neighbour(adata_section, key = coord_keys[2])

# %%
from banksy.initialize_banksy import initialize_banksy

plt.style.use('default')

banksy_dict = initialize_banksy(
    adata_section,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=True,
    plt_nbr_weights=True,
    plt_agf_angles=False, # takes long time to plot
    plt_theta=True,
)

# %%
from banksy.embed_banksy import generate_banksy_matrix

# The following are the main hyperparameters for BANKSY
lambda_list = [0.6] # list of lambda parameters

banksy_dict, banksy_matrix = generate_banksy_matrix(adata_section, banksy_dict, lambda_list, max_m)

# %%
from banksy.main import concatenate_all

banksy_dict["nonspatial"] = {
    # Here we simply append the nonspatial matrix (adata.X) to obtain the nonspatial clustering results
    0.0: {"adata": concatenate_all([adata_section.X], 0, adata=adata_section), }
}

print(banksy_dict['nonspatial'][0.0]['adata'])

# %%
## Define hyperparameters

resolutions = [0.1] # clustering resolution for UMAP
pca_dims = [20] # Dimensionality in which PCA reduces to

# %%
from banksy_utils.umap_pca import pca_umap

pca_umap(banksy_dict,
         pca_dims = pca_dims,
         add_umap = True,
         plt_remaining_var = False,
         )

# %%
from banksy.cluster_methods import run_Leiden_partition
seed = 0
results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn = 50,
    num_iterations = -1,
    partition_seed = seed,
    match_labels = True,
)

# %%
from banksy.plot_banksy import plot_results
import time

c_map =  'tab20' # specify color map
weights_graph =  banksy_dict['scaled_gaussian']['weights'][0]

# %%
banksy_path = f'./outputs/banksy_output/' 

plot_results(
    results_df,
    weights_graph,
    c_map,
    match_labels = True,
    coord_keys = coord_keys,
    max_num_labels  =  max_num_labels, 
    save_path = os.path.join(banksy_path, 'tmp_png'),
    save_fig = True, # save the spatial map of all clusters
    save_seperate_fig = True, # save the figure of all clusters plotted seperately
)

# %%
def plot_sd_vs_cell_type_composition(res_df,idx):
    """
    Plots the cell type composition as a percentage across different SD (standard deviation) values.
    The data is visualized as a stacked bar plot.

    Parameters:
    - results_df_lambda05: DataFrame containing the data with columns 'labels_scaled_gaussian_pc20_nc0.50_r0.10', 'class', and others.
    - idx: string of column of interest in the DataFrame
    
    Returns:
    - None
    """
    # Step 1: Add a 'Count' column to facilitate pivoting (each row contributes a count of 1)
    res_df.obs['Count'] = 1

    # Step 2: Create a pivot table with SD as the index, cell types as columns, and the sum of counts as values
    pivot_df = res_df.obs.pivot_table(
        index=idx,  # Group by SD
        columns='cell_types',  # Columns represent cell types
        values='Count',  # Aggregate the 'Count' column
        aggfunc='sum',  # Sum up counts for each combination
        fill_value=0  # Fill missing combinations with 0
    )

    # Ensure SD values are numeric
    pivot_df.index = pivot_df.index.astype(float)

    # Step 3: Convert counts to percentages for each SD
    # Divide each row by the row sum to get percentages, then multiply by 100
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # Step 4: Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Define figure size

    # Plot stacked bars
    bottom = None  # Keeps track of the cumulative height of the bars
    for cell_type in pivot_df.columns:  # Loop through each cell type
        ax.bar(
            pivot_df.index,  # X-axis: SD values
            pivot_df[cell_type],  # Y-axis: Percentages for this cell type
            label=cell_type,  # Legend label
            bottom=bottom  # Stack on top of previous bars
        )
        # Update 'bottom' to include the current cell type's values
        bottom = pivot_df[cell_type] if bottom is None else bottom + pivot_df[cell_type]

    # Step 5: Add labels and title
    ax.set_xlabel('SD')  # Label for the x-axis
    ax.set_ylabel('Cell Type Composition (%)')  # Label for the y-axis
    ax.set_title('SD vs. Cell Type Composition')  # Title of the plot
    ax.set_ylim(0, 100)  # Set y-axis limits to [0, 100] to represent percentages

    # Add legend
    plt.legend(
        title="Cell Type",  # Title of the legend
        bbox_to_anchor=(1.05, 1),  # Position the legend outside the plot
        loc='upper left'  # Align the legend at the upper left corner
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Step 6: Show the plot
    plt.show()

# %%
results_df

# %%
results_df.loc[idx]['adata']

# %%
idx='scaled_gaussian_pc20_nc0.60_r0.10'
results_df_lambda05 = results_df.loc[idx]['adata']
label_idx = f'labels_{idx}'
plot_sd_vs_cell_type_composition(results_df_lambda05, label_idx)

# %%


# %%

