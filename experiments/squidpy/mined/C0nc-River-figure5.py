# mined from: https://github.com/C0nc/River/blob/6ee8f8e5f46a7f6127ec18a2c37b195b059fb8b7/figure5.ipynb
# symbols: squidpy.gr.spatial_neighbors, squidpy.pl.spatial_scatter

# %%
import pysodb 
import scanpy as sc
import anndata as ad

from collections import defaultdict

sodb = pysodb.SODB()

adata_list = []
feature_list = []
label_list = []
label = []

experiment_list = sodb.list_experiment_by_dataset('chen2021dissecting')

adata_list = []


for experiment in experiment_list:
    adata = sodb.load_experiment('chen2021dissecting', experiment)
    adata = sc.pp.subsample(adata, n_obs=10000, random_state=0, copy=True)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    if 'WT' in experiment:
        adata.obs['disease'] = 'WT'
    else:
        adata.obs['disease'] = 'Diabetes'
    adata_list.append(adata)

#ad.concat(adata_list, label='slice_id').write_h5ad('o_adata.h5ad')

# %%
from river.river import River
from river.preprocess import do_slat_pair, get_the_feature, seed_everything
from collections import defaultdict
import numpy as np

seed_everything(42)

matching_list = []

for i, adata in enumerate(adata_list[1:]):
    _, best =  do_slat_pair(adata_list[0], adata, feature='pca')
    matching_list.append(best[0])
gene_expression, spatial, y, overlap = get_the_feature(adata_list, matching_list, label_key='disease')
model = River(gene_expression=gene_expression, spatial=spatial, label=y)
model.train(epoch=100)
ig_attribution, dl_attribution, sl_attribution = model.attribution()
model.summary_attribution(ig_attribution, dl_attribution, sl_attribution, overlap)
model.return_top_k_gene(top_k=200)

np.save('final_rank_diabete', model.final_rank.index.values)
np.save('ig_diabete', model.scores_ig)
np.save('dl_diabete', model.scores_dl)
np.save('sl_diabete', model.scores_sl)

# %%

import gseapy as gp
import pandas as pd
import numpy as np

# Load your ranked gene list

final_ranks = np.load('final_rank_diabete.npy', allow_pickle=True)

gene_list = final_ranks[:50].tolist()

#final_ranks = np.load('final_rank_temporal.npy', allow_pickle=True)

#final_ranks_2 = np.load('final_rank_temporal_binary.npy', allow_pickle=True)

#gene_list = list(set(final_ranks_2[:50]) - set(final_ranks[:50]))


#gene_list = final_ranks[:20].tolist()

# Define the reference gene set

for gene_sets in ['Jensen_TISSUES', 'Elsevier_Pathway_Collection', 'KEGG_2021_Human']:

    #gene_sets  = 'KEGG_2019_Mouse'

    # Run GSEA using Prerank
    enr = gp.enrich(gene_list=gene_list, # or gene_list=glist
                    gene_sets=gene_sets, # kegg is a dict object
                    background=None, # or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
                    outdir=None,
                    verbose=True)
    # Accessing the results
    results = enr.res2d
    print(results.head())
    results.to_csv('gsea_kegg_mouse_results.csv')

    # Plotting a Dotplot of the GSEA results
    gp.dotplot(results, title='GSEA Dotplot', cutoff=0.05, # Adjust the significance cutoff as needed
            color_bar_label='Normalized Enrichment Score (NES)', size=20)

    # Plotting a Barplot of the GSEA results
    gp.barplot(results, title='GSEA Barplot', cutoff=0.05, # Adjust the significance cutoff as needed
            color_bar_label='Normalized Enrichment Score (NES)')

# %%

import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import matplotlib.pyplot as plt



def cellcharter(adata):

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts", 
        batch_key='slice_id'
    )

    model = scvi.model.SCVI(adata)

    model.train(early_stopping=True, enable_progress_bar=True)
    sq.gr.spatial_neighbors(adata, spatial_key='spatial', library_key='slice_id')
    cc.gr.remove_long_links(adata)
    adata.obsm['X_scVI'] = model.get_latent_representation(adata).astype(np.float32)
    cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter', sample_key='slice_id')
    autok = cc.tl.ClusterAutoK(
        n_clusters=(1,10), 
        max_runs=10, 
        model_params=dict(
            random_state=42,
            # If running on GPU
            trainer_params=dict(accelerator='gpu', devices=1)
        )
    )

    autok.fit(adata, use_rep='X_cellcharter')
    adata.obs['cluster_cellcharter'] = autok.predict(adata, use_rep='X_cellcharter')

    '''sq.pl.spatial_scatter(
        adata, 
        color=['cluster_cellcharter'], 
        size=100, 
        shape=None,
        spatial_key='spatial',
        palette='Set2',
        figsize=(15,15),
        ncols=1
    )'''

    return adata

# %%

import anndata as ad 

adata = ad.concat(adata_list)

remaining_set = np.load('final_rank_diabete.npy', allow_pickle=True)[:200]

res_1 = cellcharter(adata[:, remaining_set].copy())
#res_2 =  cellcharter(adata)

res_1.write_h5ad('cc_200.h5ad')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc 
from matplotlib.patches import Polygon
# Example DataFrame


plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20  # Adjust title size
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Helper function to create stacked bars

adata = sc.read('cc_200.h5ad')

palette = ['#CA9C91', '#8D5FA3', '#7E594D', '#CE8DAC', '#C24A7A', '#D2AD50', '#83B756',
           '#95D1D7', '#748EBB', '#CC625F', '#FFD377', '#FD9BA0', '#BE9E33', '#C0E56F']


d = [palette[i] for i, c  in enumerate(adata.obs['ct_name'].cat.categories)]


# Helper function to create stacked bars
def create_stacked_bars(ax, proportions, offset, colors, edge_color=None):
    cumulative_proportions = np.cumsum([0] + list(proportions[:-1]))
    bars = []
    
    for i, proportion in enumerate(proportions):
        bars.append(ax.bar(offset, proportion, bottom=cumulative_proportions[i], color=colors[i % len(colors)], width=0.3, edgecolor=edge_color, linewidth=1))
    return bars


def bar(cell_type=None):
    
    adata = sc.read('cc_200.h5ad')
    if cell_type != None: 
        adata = adata[(adata.obs['slice_id'] == cell_type)].copy()

    data = adata.obs
    df = pd.DataFrame(data)

    if cell_type != None:
        proportions = df.groupby(['cluster_cellcharter', 'ct_name']).size().unstack(fill_value=0)
    else:
        proportions = df.groupby(['cluster_cellcharter', 'ct_name']).size().unstack(fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)
    '''if cell_type == None:
        new_index = ['Micro', 'Astro', 'Oligo'] + list(set(proportions.columns) - set(['Micro', 'Astro', 'Oligo']))
        proportions = proportions.reindex(columns=new_index)'''
    # Ensure proportions are calculated correctly
    proportions_1 = proportions.loc[0].values
    proportions_2 = proportions.loc[1].values
    categories = proportions.columns.tolist()

    # Colors for each category
    colors = d

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 10))


    # Create the first stacked bar plot
    bars1 = create_stacked_bars(ax, proportions_1, 0, colors, edge_color='black')

    # Create the second stacked bar plot
    bars2 = create_stacked_bars(ax, proportions_2, 1, colors, edge_color='black')
        

    # Add connections between corresponding segments with matching colors
    for i in range(len(proportions_1)):
        y1_top = np.sum(proportions_1[:i+1])
        y1_bottom = y1_top - proportions_1[i]
        y2_top = np.sum(proportions_2[:i+1])
        y2_bottom = y2_top - proportions_2[i]

        # Create a polygon to fill the area between the bars
        polygon = Polygon([(0.15, y1_bottom), (0.15, y1_top), (0.85, y2_top), (0.85, y2_bottom)], closed=True, color=colors[i % len(colors)], alpha=0.3)
        ax.add_patch(polygon)
        
        # Change the style for specific categories
        linestyle = '-'  # Dashed line for other categories
        linewidth = 1  # Thinner line for other categories
        
        # Connect bottom to bottom and top to top with matching colors
        ax.plot([0.15, 0.85], [y1_bottom, y2_bottom], color='black', linestyle=linestyle, linewidth=linewidth)
        ax.plot([0.15, 0.85], [y1_top, y2_top], color='black', linestyle=linestyle, linewidth=linewidth)

    # Set labels and title
    ax.set_ylabel('Proportion')
    ax.set_title('Change in Proportions between Two Conditions')
    ax.set_xticks([0, 1])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.set_xticklabels([0, 1], fontsize=20)

    # Add legend
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors[:len(categories)]]
    ax.legend(legend_handles, categories, loc='upper right')

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.tight_layout()
    plt.show()



bar('0')
bar('1')
bar('2')
bar('3')
bar('4')
bar('5')

# %%

import matplotlib.pyplot as plt
import squidpy as sq
import scanpy as sc
from matplotlib.colors import  ListedColormap
from palettable.cmocean.diverging import Delta_20, Balance_20
import anndata as ad
import numpy as np

adata = sc.read('cc_200.h5ad')

our_genes = np.load('final_rank_diabete.npy', allow_pickle=True)[:3]


palette = ['#CA9C91', '#8D5FA3', '#7E594D', '#CE8DAC', '#C24A7A', '#D2AD50', '#83B756',
           '#95D1D7', '#748EBB', '#CC625F', '#FFD377', '#FD9BA0', '#BE9E33', '#C0E56F']


d = [palette[i] for i, c  in enumerate(adata.obs['ct_name'].cat.categories)]

for gene in our_genes:
    
    plot_adata = adata

    norm = plt.Normalize(plot_adata[:,gene].X.min(), plot_adata[:, gene].X.max())


    cmap = ListedColormap(Balance_20.mpl_colors)

    axes = sq.pl.spatial_scatter(plot_adata, shape=None, color=[gene], figsize=(4,4), ncols=1, colorbar=False, return_ax=True, norm=norm, cmap=cmap, library_key='slice_id')
    
    
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])




plot_adata = adata



axes = sq.pl.spatial_scatter(plot_adata, shape=None, color=['ct_name'], figsize=(4,4), ncols=1, return_ax=True,library_key='slice_id', palette=ListedColormap(d))


for ax in axes:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    
    
plot_adata = adata

res_palette = ['#EC748B', '#6BB952']

axes = sq.pl.spatial_scatter(plot_adata, shape=None, color=['cluster_cellcharter'], figsize=(4,4), ncols=1, return_ax=True,  library_key='slice_id', palette=ListedColormap(res_palette))


for ax in axes:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
