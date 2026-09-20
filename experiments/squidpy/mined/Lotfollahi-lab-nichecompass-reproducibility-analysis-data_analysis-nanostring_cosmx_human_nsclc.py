# mined from: https://github.com/Lotfollahi-lab/nichecompass-reproducibility/blob/964a8ebea277dfb09004ab3d592c21b70362fad9/analysis/data_analysis/nanostring_cosmx_human_nsclc.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
sys.path.append("../../utils")

# %%
import gc
import glob
import os
import shutil
import warnings

import altair as alt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, to_rgb, to_hex
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from nichecompass.models import NicheCompass

# %%
dataset = "nanostring_cosmx_human_nsclc"

# %%
# AnnData keys
gp_names_key = "nichecompass_gp_names"
active_gp_names_key = "nichecompass_active_gp_names"
latent_key = "nichecompass_latent"

# %%
load_timestamps = ["19102023_172844_43",
                   "19102023_172844_43_3",
                   "19102023_172844_43_8"]
model_labels = ["reference",
                "reference_query",
                "reference_query"]

sample_key = "batch"

# %%
sc.set_figure_params(figsize=(6, 6))
sc.settings.set_figure_params(dpi=300)
sns.set_style("whitegrid", {'axes.grid' : False})

# %%
# Ignore future warnings and user warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# %%
plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 5
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'

# %%
# Define paths
base_path = '/lustre/groups/aih/sebastian.birk/workspace/projects/nichecompass-reproducibility'
# base_path = '/lustre/groups/imm01/workspace/irene.bonafonte/Projects/2023May_nichecompass/nichecompass-reproducibility'

# %%
def load_adata(load_timestamp, model_label='reference_query', dataset='nanostring_cosmx_human_nsclc'):
    model_folder_path = f'{base_path}/artifacts/{dataset}/models/{model_label}/{load_timestamp}'
    adata_path = f"{model_folder_path}/{dataset}_{model_label}.h5ad"
    figure_folder_path = f'{base_path}/artifacts/{dataset}/figures/{model_label}/{load_timestamp}'
    result_folder_path = f'{base_path}/artifacts/{dataset}/results/{model_label}/{load_timestamp}'
    
    adata = sc.read_h5ad(adata_path)

    batch_colors = np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('tab20b').colors)[[0,1,3,5,8,11,13,17],:])
    batch_colors = {b: c for b, c in zip(['lung5_rep1','lung5_rep2','lung5_rep3','lung6','lung9_rep1','lung9_rep2','lung12','lung13'], batch_colors)}
    adata.uns['batch_colors'] = [batch_colors[b] for b in adata.obs.batch.unique()]
    
    os.makedirs(figure_folder_path, exist_ok=True)
    os.makedirs(result_folder_path, exist_ok=True)
    
    return adata, adata_path, model_folder_path, figure_folder_path, result_folder_path

# %%
def plot_cluster_proportions(cluster_props, 
                             cluster_palette=None,
                             xlabel_rotation=0,
                             figsize=(9,4),
                             ax=None,
                             figs=None,
                             remove_legend=True): 
    if ax is None:
        figs, ax = plt.subplots(figsize=figsize)
        figs.patch.set_facecolor("white")
        figs.tight_layout()
        
    cmap = None
    if cluster_palette is not None:
        cmap = sns.palettes.blend_palette(
            cluster_palette, 
            n_colors=len(cluster_palette), 
            as_cmap=True)    
   
    cluster_props.plot(
        kind="bar", 
        stacked=True, 
        ax=ax, 
        legend=None, 
        colormap=cmap
    )
    
    if remove_legend:
        ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster").remove()
    else:
        ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    ax.tick_params(axis="x", rotation=xlabel_rotation, bottom=False)
    ax.tick_params(axis="y", rotation=90)
    ax.set_xlabel('Niche', fontsize=20)
    ax.set_ylabel("Proportion", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.spines.left.set_bounds(0, 100)
    ax.spines.right.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.top.set_visible(False)

    return ax

# %%
def plot_latent(adata, model='reference_query_mapping'):
    
    # plot umap and spatial with general characteristics
    if model == 'reference_query_mapping':
        query = adata.obs.batch[adata.obs.mapping_entity=='query'].unique()[0]
        sc.pl.umap(adata, color=['mapping_entity','cell type', 'batch','niche'], ncols=4, wspace=0.5, size=0.5)
        sc.pl.embedding(adata[adata.obs.batch==query], basis="spatial", color=['cell type', 'niche'], ncols=2, wspace=0.5, size=1)
        
    elif model == 'reference':
        sc.pl.umap(adata, color=['batch','cell type','niche'], ncols=3, wspace=0.5, size=0.5)

    # plot clusters
    cluster_res = adata.obs.columns[adata.obs.columns.str.contains('latent_leiden_')]
    for cl in cluster_res:
        if f'{cl}_colors' in adata.uns.keys():
            del adata.uns[f'{cl}_colors']
    sc.pl.umap(adata, color=cluster_res, ncols=4, wspace=0.5, size=0.5, palette=sc.pl.palettes.vega_20_scanpy)
    
    if model == 'reference_query_mapping':
        sc.pl.embedding(adata[adata.obs.batch==query], basis="spatial", color=cluster_res, ncols=4, wspace=0.5, size=1)    
        
    return

# %%
def res_details(adata, resolution=0.4, model='reference_query_mapping'):

    # plot spatial for all samples
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(21, 6))
    for i, batch in enumerate(adata.obs.batch.unique()):
        if i < 4:
            sc.pl.embedding(adata[adata.obs.batch==batch], basis="spatial", color=f'latent_leiden_{resolution}', size=1, show=False, ax=axes[0,i], title=batch)
        else:
            sc.pl.embedding(adata[adata.obs.batch==batch], basis="spatial", color=f'latent_leiden_{resolution}', size=1, show=False, ax=axes[1,i-4], title=batch)
    plt.tight_layout()
    plt.show()
    
    # plot proportions
    figs, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 3))
    plot_var = f'latent_leiden_{resolution}'
    
    for i, cluster_var in enumerate(['cell type', 'niche', 'batch']):
        props = adata.obs.groupby([cluster_var, plot_var]).size().reset_index()
        props = props.pivot(columns=plot_var, index=cluster_var).T
        props.index = props.index.droplevel(0)
        props.fillna(0, inplace=True)
        props = props.div(props.sum(axis=1), axis=0)*100 
        axes[i] = plot_cluster_proportions(props, xlabel_rotation=90, cluster_palette=adata.uns[f'{cluster_var}_colors'], figsize=(4,3), ax=axes[i], figs=figs)
    figs.show()

    return

# %%
def colorFader(c1, c2='#FFFFFF', n=10, mix=0):
    n+=1
    c1=np.array(to_rgb(c1))
    c2=np.array(to_rgb(c2))
    colors=[]
    for x in range(n+1):
         colors.append(to_hex((1-x/n)*c1 + c2*x/n))
    return colors[:-1]

# %%
adata, \
adata_path, \
model_folder_path, \
figure_folder_path, \
result_folder_path = load_adata(load_timestamp=load_timestamps[0],
                                model_label=model_labels[0])

# %%
latent_leiden_resolution = 0.45
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"

# %%
# Merge super small cluster
adata.obs.loc[(adata.obs['latent_leiden_0.45']=='12').values,'latent_leiden_0.45'] = '7'
adata.obs['latent_leiden_0.45'] = adata.obs['latent_leiden_0.45'].cat.remove_unused_categories()

# %%
plot_latent(adata, model='reference')

# %%
res_details(adata, resolution=latent_leiden_resolution, model='reference')

# %%
# Label niches
leiden2niche = {
    '0': '1- Tumor (stroma border)', '2': '2- Tumor interior', '5': '3- Tumor (neutrophil border)', '7': '4- Tumor interior', '11': '5- Infiltrated tumor',
    '3': '6- Tumor-infiltrating neutrophils', '4': '7- Neutrophil expansion', '1': '8- Stroma', '6': '9- Plasmablast rich stroma', '9': '10- Lymphoid rich stroma', '8': '11- Lymphoid aggregates', '10': '12- Macrophage rich stroma'
}

leiden2leiden = {'0': '1', '2': '2', '5': '3', '7': '4', '11': '5', '3': '6', '4': '7', '1': '8', '6': '9', '9': '10', '8': '11', '10': '12'} # numbers based on dendogram order (see below)
adata.obs['niche'] = adata.obs['latent_leiden_0.45'].map(leiden2niche)
adata.obs['leidenOrd'] = adata.obs['latent_leiden_0.45'].map(leiden2leiden)

# Format donor names
adata.obs['donor'] = adata.obs.batch.str.replace('_',' ')
adata.obs['donor'] = adata.obs.donor.str.replace('lung','Donor ')
adata.obs['donor'] = adata.obs.donor.str.replace('rep','r')
adata.uns['donor_colors'] = adata.uns['batch_colors']

# %%
# Group niches
if not 'cluster_grups' in adata.uns:
    adata.uns['cluster_groups'] = {}
    
adata.uns['cluster_groups'][f"latent_leiden_0.45"] = {
    'tumor_clusters': ['0', '2', '5', '7', '11'],
    'stroma_clusters': ['3', '4', '1', '6', '9', '8', '10'],
    'neutrophil_clusters': ['3', '4'],
    'macrophage_clusters': ['10'],
    'lymphoid_clusters': ['6', '9', '8']
}

adata.uns['cluster_groups'][f"leidenOrd"] = {
    'tumor_clusters': ['1', '2', '3', '4', '5'],
    'stroma_clusters': ['6', '7', '8', '9', '10', '11', '12'],
    'neutrophil_clusters': ['6', '7'],
    'macrophage_clusters': ['12'],
    'lymphoid_clusters': ['9', '10', '11']
}

# %%
### Extended Data Fig. 5a ###
sc.tl.dendrogram(adata=adata,
                 use_rep="nichecompass_latent",
                 n_pcs=adata.obsm['nichecompass_latent'].shape[1],
                 groupby="niche")
fig, (ax) = plt.subplots(1, 1, figsize=(0.6, 1.5))
sc.pl.dendrogram(
    adata=adata,
    groupby="niche",
    orientation="left",
    ax=ax,
    save="_nichecompass_latent.svg")
plt.show()
if os.path.exists(f"{figure_folder_path}/dendrogram_nichecompass_latent.svg"):
    os.remove(f"{figure_folder_path}/dendrogram_nichecompass_latent.svg")
shutil.move(f"./figures/dendrogram_nichecompass_latent.svg", figure_folder_path)
shutil.rmtree("./figures/")

# %%
# Set niche colors
rcParams['figure.figsize'] = (4, 0.5)
general=np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('Dark2').colors))
a=np.outer(np.ones(len(general)),np.arange(0,1,0.01))   # pseudo image data
plt.imshow(a,aspect='auto',cmap=plt.get_cmap('Dark2'),origin="lower")
rcParams['figure.figsize'] = (4, 3)

# use dendogram order to define niche palette related to cell type
general=np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('Dark2').colors))
tumor=colorFader(general[2], c2='#FFFFFF', n=5, mix=0)
lymphoid=general[4]
myeloid=general[1]
Blike=colorFader(general[0], c2='#FFFFFF', n=2, mix=0)
neutrophil=colorFader(general[3], c2='#FFFFFF', n=2, mix=0)
stroma=general[5]

leiden_colDict = {
    '0': tumor[0], '2': tumor[1], '5': tumor[2], '7': tumor[3], '11': tumor[4], 
    '3': neutrophil[0], '4': neutrophil[1], 
    '1': stroma, '6': Blike[0], '9': Blike[1], '8': lymphoid, '10': myeloid 
}
adata.uns['latent_leiden_0.45_colors'] = [x for x in leiden_colDict.values()]
adata.uns['leidenOrd_colors'] = [x for x in leiden_colDict.values()]
adata.uns['niche_colors'] = [x for x in leiden_colDict.values()]

# %%
### Fig. 5a ###
rcParams['figure.figsize'] = (8, 6)
sc.pl.umap(adata,
           color=['niche'],
           ncols=1,
           wspace=1,
           size=2.5,
           legend_fontsize='x-large',
           show=False,
           frameon=False,
           title=[''])
plt.savefig(f"{figure_folder_path}/5_a.svg", bbox_inches="tight", format='svg')
sc.pl.umap(adata, color=['leidenOrd'], ncols=1, wspace=1, size=2.5, legend_fontsize='x-large', show=False, frameon=False, title=[''], legend_loc='on data')
plt.savefig(f"{figure_folder_path}/5_a_num.svg", bbox_inches="tight", format='svg')
rcParams['figure.figsize'] = (4, 3)

# %%
### Fig. 5b ###
rcParams['figure.figsize'] = (8, 6)
sc.pl.umap(adata, color=['cell type'], ncols=1, wspace=1, size=2.5, legend_fontsize='x-large', show=False, frameon=False, title=[''])
plt.savefig(f"{figure_folder_path}/5_b.svg", bbox_inches="tight", format='svg')
rcParams['figure.figsize'] = (4, 3)

# %%
### Fig. 5c ###
rcParams['figure.figsize'] = (8, 6)

sc.pl.umap(adata[np.random.choice(adata.n_obs, adata.n_obs, replace=False)], # reshufle adata so that cells are not ordered by donor in the plot
           color=['donor'],
           ncols=1,
           wspace=1,
           size=2.5,
           legend_fontsize='x-large',
           show=False,
           frameon=False,
           title=[''])
plt.savefig(f"{figure_folder_path}/5_c.svg", bbox_inches="tight", format='svg')
rcParams['figure.figsize'] = (4, 3)

# %%
### Extended Data Fig. 5e ###
n = adata.obs.batch.nunique()
fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(4*n,3*2))

for i, batch in enumerate(adata.obs.batch.unique()):
    ax = sc.pl.embedding(adata[adata.obs.batch==batch],
                         basis="spatial",
                         color=['cell type'],
                         size=6,
                         legend_loc=None,
                         frameon=False,
                         title=[''],
                         ax=axes[0,i],
                         show=False)
    ax = sc.pl.embedding(adata[adata.obs.batch==batch], basis="spatial", color=['leidenOrd'], size=6, legend_loc=None, frameon=False, title=[''], ax=axes[1,i], show=False)

fig.tight_layout()
fig.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig(f"{figure_folder_path}/e5_e.svg", bbox_inches="tight", format='svg')

# %%
### Fig. 5d ###
# add gp
gp = 'CXCL1_ligand_receptor_GP'
adata.obs[gp] = - adata.obsm['nichecompass_latent'][:,adata.uns['nichecompass_active_gp_names']==gp]

rcParams['figure.figsize'] = (4, 3)
batches = ['lung9_rep2','lung12']
nr = len(batches)
nc = 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(4*nc,3*nr))
for i, batch in enumerate(batches):
    ax = sc.pl.embedding(adata[adata.obs.batch==batch],
                         basis="spatial",
                         color=['cell type'],
                         size=6,
                         frameon=False,
                         title=[''],
                         ax=axes[i,1],
                         show=False,
                         legend_loc=None)
    ax = sc.pl.embedding(adata[adata.obs.batch==batch],
                         basis="spatial", color=['niche'],
                         size=6,
                         frameon=False,
                         title=[''],
                         ax=axes[i,0],
                         show=False,
                         legend_loc=None,
                         groups=['1- Tumor (stroma border)','3- Tumor (neutrophil border)','6- Tumor-infiltrating neutrophils'])
    ax = sc.pl.embedding(adata[adata.obs.batch==batch],
                         basis="spatial",
                         color=['CXCL1_ligand_receptor_GP'],
                         size=10,
                         frameon=False,
                         title=[''],
                         ax=axes[i,2],
                         show=False,
                         colorbar_loc=None,
                         cmap='RdGy_r')
    
fig.tight_layout()
fig.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig(f"{figure_folder_path}/5_d.svg", bbox_inches="tight", format='svg')

# %%
### Supplementary Fig. 28a ###
rcParams['figure.figsize'] = (2, 3)
adata_temp = adata.obs[(adata.obs.patient == 'Lung9') & (adata.obs.leidenOrd.isin(['1','3']))]
adata_temp['leidenOrd'] = adata_temp.leidenOrd.cat.remove_unused_categories()
sns.boxplot(data=adata_temp, x='leidenOrd', y='CXCL1_ligand_receptor_GP', palette=adata.uns['leidenOrd_colors'], showfliers = False).set(title='')
plt.ylim((-75, 100))
plt.xticks(rotation=0)
plt.xlabel("", fontsize=17.5)
plt.ylabel('CXCL1 LR GP', fontsize=17.5)
plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.7), fontsize=15)
plt.tick_params(bottom=False, labelsize=17.5)
sns.despine(offset=10, trim=True, bottom=True)
rcParams['figure.figsize'] = (4, 3)
plt.savefig(f"{figure_folder_path}/s28_a.svg", bbox_inches="tight", format='svg')

# Get number of cells for figure caption
print(len(adata_temp[adata_temp['leidenOrd'] == '1']))
print(len(adata_temp[adata_temp['leidenOrd'] == '3']))
del adata_temp

# %%
### Fig. 5e ###
niches_list = (
    ['9- Plasmablast rich stroma','11- Lymphoid aggregates'],
    ['6- Tumor-infiltrating neutrophils','7- Neutrophil expansion'],
    ['6- Tumor-infiltrating neutrophils','7- Neutrophil expansion','9- Plasmablast rich stroma','11- Lymphoid aggregates']
)

batch_list = (
    ['lung5_rep1'], # highlight lymphoid niches
    ['lung5_rep1','lung9_rep2'], # highlight neutrophil niches
    ['lung12'] # donor 12 has a few cells in shared niches in both comparments, and we highlight all for comparison
)

for nfig, (batches, niches) in enumerate(zip(batch_list, niches_list)):
    n = len(batches)
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(4*n,3*2))
    for i, batch in enumerate(batches):
        if len(batches) == 1:
            ax1 = axes[1]
            ax0 = axes[0]
        else:
            ax1 = axes[1,i]
            ax0 = axes[0,i]
            
        ax = sc.pl.embedding(adata[adata.obs.batch==batch],
                             basis="spatial",
                             color=['niche'],
                             size=6,
                             groups=niches,
                             frameon=False,
                             title=[''],
                             ax=ax0,
                             show=False,
                             legend_loc=None)
        ax = sc.pl.embedding(adata[adata.obs.batch==batch], basis="spatial", size=3, frameon=False, title=[''], ax=ax1, show=False, legend_loc=None) # gray background
        ax = sc.pl.embedding(adata[(adata.obs.batch==batch) & adata.obs.niche.isin(niches)],
                             basis="spatial",
                             color=['cell type'],
                             size=6,
                             frameon=False,
                             title=[''],
                             ax=ax1,
                             show=False,
                             legend_loc=None)


    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(f"{figure_folder_path}/5_e{nfig+1}.svg", bbox_inches="tight", format='svg')    

# %%
### Extended Data Fig. 5c ###
cluster_var = 'cell type'
plot_var = 'leidenOrd'
props = adata.obs.groupby([cluster_var, plot_var]).size().reset_index()
props = props.pivot(columns=plot_var, index=cluster_var).T
props.index = props.index.droplevel(0)
props.fillna(0, inplace=True)
props = props.div(props.sum(axis=1), axis=0)*100 
fig = plot_cluster_proportions(props,
                               xlabel_rotation=90,
                               cluster_palette=adata.uns[f'{cluster_var}_colors'],
                               figsize=(8,3),
                               remove_legend=False)
plt.savefig(f"{figure_folder_path}/e5_c.svg", bbox_inches="tight", format='svg')

# %%
### Extended Data Fig. 5d ###
cluster_var = 'donor'
plot_var = 'leidenOrd'
props = adata.obs.groupby([cluster_var, plot_var]).size().reset_index()
props = props.pivot(columns=plot_var, index=cluster_var).T
props.index = props.index.droplevel(0)
props.fillna(0, inplace=True)
props = props.div(props.sum(axis=1), axis=0)*100 
fig = plot_cluster_proportions(props,
                               xlabel_rotation=90,
                               cluster_palette=adata.uns[f'{cluster_var}_colors'],
                               figsize=(8,3),
                               remove_legend=False)
plt.savefig(f"{figure_folder_path}/e5_d.svg", bbox_inches="tight", format='svg')

# %%
### Supplementary Fig. 28b ###
cluster_var = 'donor'
plot_var = 'leidenOrd'
props = adata[adata.obs.leidenOrd.isin(['3','5'])].obs.groupby([cluster_var, plot_var]).size().reset_index()
props = props.pivot(columns=plot_var, index=cluster_var).T
props.index = props.index.droplevel(0)
props.fillna(0, inplace=True)
props = props.div(props.sum(axis=1), axis=0)*100 
fig = plot_cluster_proportions(props,
                               xlabel_rotation=90,
                               cluster_palette=adata.uns[f'{cluster_var}_colors'],
                               figsize=(2,3),
                               remove_legend=False)
plt.savefig(f"{figure_folder_path}/s28_b.svg", bbox_inches="tight", format='svg')

# %%
### Extended Data Fig. 5f ###
rcParams['figure.figsize'] = (8, 12)
adata_tumor_subset = adata[(adata.obs['niche'].str.contains("umor")) & (~adata.obs['niche'].str.contains("neutrophils"))]

sc.tl.dendrogram(adata=adata_tumor_subset,
                 use_rep="nichecompass_latent",
                 n_pcs=adata.obsm['nichecompass_latent'].shape[1],
                 groupby="niche")

sc.tl.rank_genes_groups(adata_tumor_subset,
                        'niche')
sc.set_figure_params(fontsize=12, dpi=300)
sc.pl.rank_genes_groups_dotplot(adata_tumor_subset, n_genes=5, show=False)
plt.savefig(f"{figure_folder_path}/e5_f.svg", bbox_inches="tight", format='svg')
rcParams['figure.figsize'] = (4, 3)

# %%
### Extended Data Fig. 6a ###
adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X

# cell types markers
markers = {
    'B-cell': ['MS4A1', 'CD37', 'CD79A','CD19'],
    'NK/T cell': ['CD2','IL7R','CD3G','CTLA4','CD69','GZMA','CD28','TIGIT'],
    'endothelial': ['VWF','FLT1','CDH5','CLEC14A','RAMP2'],
    'epithelial': ['CCL20','LAMP3','AQP3'],
    'fibroblast': ['COL1A1','COL3A1','FN1'],
    'mast': ['TPSB2', 'TPSAB1', 'CPA3'],
    'DC/monocyte': ['CD163','CD74','HLA-DRB1','LYZ','FCGR3A','CD68','CD14'], # DC, monocyte, macrophage (SPP1 or not)    
    'macrophage': ['MARCO', 'SPP1', 'C1QC', 'GPNMB'],
    'neutrophil': ['CXCL8','CXCR1','CXCR2','IL1R2'],
    'plasmablast': ['IGKC','IGHG1','JCHAIN','XBP1','MZB1', 'CD38'],
}
markers_list = [x for y in markers.values() for x in y]

# cell types to highlight in each niche
stroma_niches = adata.obs.loc[adata.obs.leidenOrd.isin(adata.uns['cluster_groups']['leidenOrd']['stroma_clusters']),'niche'].unique()
keep_niche = []
niche_name = []
for niche in stroma_niches:
    props = (adata[adata.obs.niche == niche].obs['cell type'].value_counts() / len(adata[adata.obs.niche == niche]))
    keep = adata[adata.obs.niche == niche].obs['cell type'].value_counts()[(adata[adata.obs.niche == niche].obs['cell type'].value_counts() / len(adata[adata.obs.niche == niche])) > 0.1].index.values
    keep_niche.append(keep.astype(str).tolist())
    niche_name.append(niche)
    
# Get cell type proportions across all stroma niches
global_stroma_props = (adata[adata.obs.niche.isin(niche_name)].obs['cell type'].value_counts() / len(adata[adata.obs.niche.isin(niche_name)]))
print(adata[adata.obs.niche.isin(niche_name)].obs['cell type'].value_counts())

# Get color norm for barplot hue
stroma_min = global_stroma_props.min()
stroma_max = global_stroma_props.max()
norm = mcolors.Normalize(vmin=stroma_min, vmax=stroma_max)

# set up grid
TEXT_WIDTH = 6.7261  # in
DPI = 360

def set_font_size(font_size):
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size-2)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size-2)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size-4, title_fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)   # fontsize of the figure title
    
sns.set(context='paper', style='whitegrid')
plt.rc('grid', linewidth=0.3)
sns.set_palette('colorblind')
sc.set_figure_params(vector_friendly=True, dpi_save=DPI)
set_font_size(10)
plt.rcParams['figure.constrained_layout.use'] = True


fig = plt.figure(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.5), dpi=DPI)
gridspecs = {}

gridspecs["columns"] = mpl.gridspec.GridSpec(
    figure=fig,
    nrows=1,
    ncols=2,
    height_ratios=[1],
    width_ratios=[1, 4],
)

gridspecs['barplots'] = mpl.gridspec.GridSpecFromSubplotSpec(
    subplot_spec=gridspecs["columns"][0],
    nrows=len(niche_name),
    ncols=1,
    height_ratios=[len(x) for x in keep_niche],
    width_ratios=[1],
    hspace=0.35,
)
gridspecs['heatmaps'] = mpl.gridspec.GridSpecFromSubplotSpec(
    subplot_spec=gridspecs["columns"][1],
    nrows=len(niche_name),
    ncols=1,
    height_ratios=[len(x) for x in keep_niche],
    width_ratios=[1],
    hspace=0.35,
)

ax_barplots, ax_heatmaps = {}, {}
for i in range(len(keep_niche)):
    ax_barplots[i] = fig.add_subplot(gridspecs['barplots'][i])
    ax_heatmaps[i] = fig.add_subplot(gridspecs['heatmaps'][i])
    

# cell type proportion
batch_col_dict = pd.Series({cell: color for cell, color in zip(adata.obs['cell type'].cat.categories, adata.uns['cell type_colors'])})
for i, niche in enumerate(niche_name):
    props = (adata[adata.obs.niche == niche].obs['cell type'].value_counts() / len(adata[adata.obs.niche == niche]))
    props = pd.DataFrame({'cell type': keep_niche[i], 'proportion': props[keep_niche[i]], 'stroma_proportion': global_stroma_props[keep_niche[i]], 'color': batch_col_dict[keep_niche[i]]})
    print(props)
    
    sns.barplot(props, x='proportion', y='cell type', hue='stroma_proportion', hue_norm=norm, ax=ax_barplots[i], dodge=False, palette='Blues' , width=0.8)
    ax_barplots[i].legend().remove()
    ax_barplots[i].tick_params(axis="x", rotation=90, bottom=False)
    ax_barplots[i].tick_params(axis="y", rotation=0)
    ax_barplots[i].set_xlabel('')
    ax_barplots[i].set_ylabel('')
    ax_barplots[i].tick_params(axis='both', which='major', labelsize=6)
    ax_barplots[i].spines.left.set_visible(False)
    ax_barplots[i].spines.right.set_visible(False)
    ax_barplots[i].spines.bottom.set_visible(False)
    ax_barplots[i].spines.top.set_visible(False)
    ax_barplots[i].set_xlim(ax_barplots[i].get_xlim()[1], ax_barplots[i].get_xlim()[0])
    ax_barplots[i].yaxis.set_label_position('right')
    ax_barplots[i].yaxis.set_ticks_position('right')
    ax_barplots[i].get_xaxis().set_visible(False)
    ax_barplots[i].tick_params(axis='y', which='both',length=0)

# cell type markers
for i, niche in enumerate(niche_name):
    adata_tmp = adata[(adata.obs.niche == niche) & adata.obs['cell type'].isin(keep_niche[i])]
    data = pd.DataFrame(adata_tmp[:,markers_list].layers['scaled'], index=adata_tmp.obs_names, columns=markers_list)
    data['cell type'] = adata_tmp.obs['cell type'].values
    data = data.groupby('cell type').mean()
    data = data.loc[keep_niche[i],:]
    
    ax_heatmaps[i] = sns.heatmap(data, cmap='RdBu_r', center=0, cbar=False, linewidth=0.1, linecolor='gray', square=False, xticklabels=False, yticklabels=False, ax=ax_heatmaps[i])        
    # ax_heatmaps[i] = sns.heatmap(data, cmap='RdBu_r', center=0, cbar=False, linewidth=0.1, linecolor='gray', square=False, xticklabels=True, yticklabels=False, ax=ax_heatmaps[i]) # to get gene labels
    ax_heatmaps[i].set_ylabel('')
    ax_heatmaps[i].set_xlabel('')
    ax_heatmaps[i].yaxis.set_label_position('right')
    ax_heatmaps[i].patch.set_edgecolor('black')
    ax_heatmaps[i].patch.set_linewidth(1)

plt.savefig(f"{figure_folder_path}/e6_a1.svg", bbox_inches="tight", format='svg')

# this plot is to get x labels
plot_dict = sc.pl.matrixplot(
    adata, markers, groupby='cell type', dendrogram=False, layer="scaled", cmap="RdBu_r", 
    vcenter=0, standard_scale=None, show=False)
ax = plot_dict['mainplot_ax']    # The axis object
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(15)
plt.savefig(f"{figure_folder_path}/e6_a2.svg", bbox_inches="tight", format='svg')

print(stroma_niches.astype(str)) # ordered labels for each row

# %%
### Extended Data Fig. 6b ###
fs=16
def plot_celltype_batch(cluster_props,
                        cluster_palette=None,
                        ax=None,
                        title=''): 
        
    cmap = None
    if cluster_palette is not None:
        cmap = sns.palettes.blend_palette(
            cluster_palette, 
            n_colors=len(cluster_palette), 
            as_cmap=True)    
   
    cluster_props.plot(
        kind="bar", 
        stacked=True, 
        ax=ax, 
        legend=None, 
        colormap=cmap
    )
    
    ax.legend().remove()
    ax.tick_params(axis="x", rotation=90, bottom=False)
    ax.tick_params(axis="y", rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel("Proportion", fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.spines.right.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    for label in ax.get_xticklabels(minor=True):
        label.set_verticalalignment('center')
    ax.set_title(title, fontsize=fs)

    return ax

def plot_counts_batch(cluster_props, 
                             cluster_palette=None,
                             ax=None): 
    cluster_props = cluster_props.reset_index().set_axis(['donor','proportion'], axis=1)   
    
    print(cluster_props)
    
    sns.barplot(cluster_props, x='donor', y='proportion', ax=ax, dodge=False, width=0.6)
    
    ax.legend().remove()
    ax.tick_params(axis="x", rotation=90, bottom=False)
    ax.tick_params(axis="y", rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel("Proportion", fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.spines.right.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_ylim(100, 0)

    return ax

niche_rename = {'8- Stroma': '8- Stroma', '7- Neutrophil expansion': '7- Neutrophil\nexpansion', '6- Tumor-infiltrating neutrophils': '6- Tumor-infiltrating\nneutrophils', 
                '9- Plasmablast rich stroma': '9- Plasmablast\nrich stroma', '12- Macrophage rich stroma': '12- Macrophage\nrich stroma', 
                '11- Lymphoid aggregates': '11- Lymphoid\naggregates', '10- Lymphoid rich stroma': '10- Lymphoid rich\nstroma'}

# order stroma niches from patient shared to patient specific
plot_var = 'patient'
cluster_var = 'niche'
props_patient = adata.obs.groupby([cluster_var, plot_var]).size().reset_index().pivot(columns=plot_var, index=cluster_var).T.fillna(0, inplace=False)
props_patient = 100 * props_patient.div(props_patient.sum(axis=0), axis=1).loc[:,props_patient.columns.isin(stroma_niches)]
niche_order = props_patient.var(axis=0).sort_values().index.values

# compute niche distribtuion per replicate
plot_var = 'donor'
props_batch = adata.obs.groupby([cluster_var, plot_var]).size().reset_index().pivot(columns=plot_var, index=cluster_var).T.fillna(0, inplace=False)
props_batch.index = props_batch.index.droplevel(0).astype(str)
props_batch = 100 * props_batch.div(props_batch.sum(axis=0), axis=1).loc[:,niche_order]

# plot
cluster_var = 'cell type'
width = np.sum(props_batch > 5, axis=0)
fig, axes = plt.subplots(nrows=2, ncols=props_batch.shape[1], figsize=(np.sum(width), 9), width_ratios=width,  height_ratios=[1,1], sharex=False, sharey='row')
plt.subplots_adjust(wspace=0.5, hspace=0.6)

for i, niche in enumerate(niche_order):
    batches_sorted = props_batch[niche].sort_values(ascending=False).index[props_batch[niche].sort_values(ascending=False) > 5]
    adata_tmp = adata[(adata.obs.niche == niche) & adata.obs.donor.isin(batches_sorted)]
    niche_props = adata_tmp.obs.groupby([cluster_var, plot_var]).size().reset_index().pivot(columns=plot_var, index=cluster_var).T.fillna(0, inplace=False)
    niche_props.index = niche_props.index.droplevel(0)
    niche_props = niche_props.div(niche_props.sum(axis=1), axis=0)*100 
    axes[0,i] = plot_celltype_batch(niche_props.loc[batches_sorted,:], cluster_palette=adata.uns[f'{cluster_var}_colors'], ax=axes[0,i], title=niche_rename[niche])
    axes[1,i] = plot_counts_batch(props_batch.loc[batches_sorted,niche], ax=axes[1,i])

fig.tight_layout(pad=5.0)
fig.show()

plt.savefig(f"{figure_folder_path}/e6_b.svg", bbox_inches="tight", format='svg')

# %%
### Supplementary Fig. 27 ###
# Compute n neighbours per cell type
for n in [4, 25, 50]: 
    knn = {}
    cell_counts = {}
    for b in adata.obs.batch.unique():
        adata_tmp = adata[adata.obs.batch==b]
        celltypes = adata_tmp.obs['cell type'].astype(str).values.astype('<U22')
        cellnames = adata_tmp.obs_names
        
        sq.gr.spatial_neighbors(adata_tmp,
                                coord_type="generic",
                                spatial_key="spatial",
                                n_neighs=n)

        # Make adjacency matrix symmetric
        adata_tmp.obsp['spatial_connectivities'] = (
            adata_tmp.obsp['spatial_connectivities'].maximum(
                adata_tmp.obsp['spatial_connectivities'].T))
        
        boolean_connectivities = adata_tmp.obsp['spatial_connectivities'].A.astype(bool)

        for i in range(len(cellnames)):
            unique, counts = np.unique(celltypes[boolean_connectivities[i,:]], return_counts=True)
            cell_counts[cellnames[i]] = dict(zip(unique, counts))

    adata.obsm[f'k{n}_neighbours_celltype'] = pd.DataFrame(cell_counts).T.fillna(0)


# Plot neighbor distribution in tumor niches
for n in [4, 25, 50]: 
    # format for plotting and keep only those with an interesting number
    leiden_col_key='leidenOrd'
    dt = adata.obsm[f'k{n}_neighbours_celltype'].copy()
    dt['leidenOrd'] = adata.obs[leiden_col_key].copy()
    dt = dt[(adata.obs['cell type'] == 'tumor') & (adata.obs.niche.str.contains('umor')) & ~(adata.obs.niche.str.contains('neutrophils'))]
    # dt = dt.loc[dt[leiden_col_key]=='tumor']
    dt[leiden_col_key] = dt[leiden_col_key].cat.remove_unused_categories()
    dt = dt.reset_index()
    dt = pd.melt(dt, id_vars=['index',leiden_col_key], value_name=f'n NN/{n}', var_name='cell type')
    dt['cell type'] = pd.Categorical(dt['cell type'], categories=adata.obs['cell type'].cat.categories)

    include = (dt.groupby(['cell type','leidenOrd'])[f'n NN/{n}'].mean().groupby(['cell type']).max() > n*0)
    # include = (dt.groupby(['cell type',leiden_col_key])[f'n NN/{n}'].mean().groupby(['cell type']).max() > n*0.15) & (dt.groupby(['cell type',leiden_col_key])[f'n NN/{n}'].mean().groupby(['cell type']).max() < n*0.6)
    # rcParams['figure.figsize'] = (adata.obs[leiden_col_key].nunique()*2, 4)
    dt = dt[dt['cell type'].isin(include[include].index.values)]
    dt['cell type'] = dt['cell type'].cat.remove_unused_categories()
    
    print(dt.head())

    # plot
    plt.figure(figsize=(20, 4))
    sns.boxplot(data=dt, x=leiden_col_key, y=f'n NN/{n}', hue='cell type', palette=adata.uns[f'cell type_colors'][include.values], showfliers = False).set(title='')
    #plt.ylim((0, n))
    plt.xticks(rotation=0)
    plt.xlabel("", fontsize=17.5)
    plt.ylabel(f'neighborhood composition', fontsize=17.5)
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.7), fontsize=15)
    plt.tick_params(bottom=False, labelsize=17.5)
    sns.despine(offset=10, trim=True, bottom=True)
    plt.savefig(f"{figure_folder_path}/s27_k{n}.svg", bbox_inches="tight", format='svg')
rcParams['figure.figsize'] = (4, 3)

# Get number of cells for figure caption
print(len(dt[(dt['leidenOrd'] == '1') & (dt['cell type'] == 'tumor')]))
print(len(dt[(dt['leidenOrd'] == '2') & (dt['cell type'] == 'tumor')]))
print(len(dt[(dt['leidenOrd'] == '3') & (dt['cell type'] == 'tumor')]))
print(len(dt[(dt['leidenOrd'] == '4') & (dt['cell type'] == 'tumor')]))
print(len(dt[(dt['leidenOrd'] == '5') & (dt['cell type'] == 'tumor')]))

# %%
### Supplementary Fig. 28c ###
# Compute spatial neighbors only sample 12 for k=25
leiden_col_key='leidenOrd'
n=25

# compute n neighbours per cell type
knn = {}
cell_counts = {}
b = 'lung12'
X = adata[adata.obs.batch==b].obsm['spatial']
celltypes = adata[adata.obs.batch==b].obs['cell type'].astype(str).values.astype('<U22')   
cellnames = adata[adata.obs.batch==b].obs_names

knn[b] = NearestNeighbors(n_neighbors=n)
knn[b].fit(X)
knn[b] = knn[b].kneighbors(X, return_distance=False)    
knn[b] = celltypes[knn[b]]

for i in range(len(cellnames)):
    unique, counts = np.unique(knn[b][i,:], return_counts=True)
    cell_counts[cellnames[i]] = dict(zip(unique, counts))

res = pd.DataFrame(cell_counts).T.fillna(0)
res[leiden_col_key] = adata.obs.loc[res.index,leiden_col_key]

structure='tumor_clusters'
groups=adata.uns['cluster_groups'][leiden_col_key][structure]
    
# format for plotting and keep only those with an interesting number
dt = res.copy()
dt = dt.loc[dt[leiden_col_key].isin(['3','5']),:]
dt[leiden_col_key] = dt[leiden_col_key].cat.remove_unused_categories()
dt = dt.reset_index()
dt = pd.melt(dt, id_vars=['index',leiden_col_key], value_name=f'n NN/{n}', var_name='cell type')
dt['cell type'] = pd.Categorical(dt['cell type'], categories=adata.obs['cell type'].cat.categories)
include = (dt.groupby(['cell type',leiden_col_key])[f'n NN/{n}'].mean().groupby(['cell type']).max() > n*0.05) & (dt.groupby(['cell type',leiden_col_key])[f'n NN/{n}'].mean().groupby(['cell type']).max() < n*0.6)
dt = dt[dt['cell type'].isin(include[include].index.values)]
dt['cell type'] = dt['cell type'].cat.remove_unused_categories()

# plot
#rcParams['figure.figsize'] = (2*1.5, 4)
plt.figure(figsize=(6, 4))
sns.boxplot(data=dt, x=leiden_col_key, y=f'n NN/{n}', hue='cell type', palette=adata.uns[f'cell type_colors'][include.values], showfliers = False).set(title='')
plt.ylim((0, 25))
plt.xticks(rotation=0)
plt.xlabel("", fontsize=17.5)
plt.ylabel(f'neighborhood composition', fontsize=17.5)
plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.7), fontsize=15)
plt.tick_params(bottom=False, labelsize=17)
sns.despine(offset=10, trim=True, bottom=True)
plt.savefig(f"{figure_folder_path}/s28_c.svg", bbox_inches="tight", format='svg')
plt.show()
rcParams['figure.figsize'] = (4, 3)

# Get number of cells for figure caption
print(len(dt[(dt['leidenOrd'] == '3') & (dt['cell type'] == 'fibroblast')]))
print(len(dt[(dt['leidenOrd'] == '3') & (dt['cell type'] == 'neutrophil')]))
print(len(dt[(dt['leidenOrd'] == '5') & (dt['cell type'] == 'fibroblast')]))
print(len(dt[(dt['leidenOrd'] == '5') & (dt['cell type'] == 'neutrophil')]))

# %%
adata, \
adata_path, \
model_folder_path, \
figure_folder_path, \
result_folder_path = load_adata(load_timestamp=load_timestamps[1],
                                model_label=model_labels[1])

# %%
plot_latent(adata, model='reference_query_mapping')

# %%
latent_leiden_resolution = 0.35
res_details(adata, resolution=latent_leiden_resolution, model='reference_query_mapping')

# %%
# Set niche colors
leiden2niche = {
    '2': '1- Tumor (stroma border)', '4': '2- Tumor interior', '8': '3- Tumor (neutrophil border)', '5': '4- Tumor interior', '11': '5- Infiltrated tumor',
    '7': '6- Tumor-infiltrating neutrophils', '6': '7- Neutrophil expansion', '10': '8- Stroma', '1': '9- Plasmablast rich stroma', '9': '10- Lymphoid rich stroma', '3': '11- Lymphoid aggregates', '10': '12- Macrophage rich stroma'
}

adata.obs['niche'] = adata.obs['latent_leiden_0.35'].map(leiden2niche)

general=np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('Dark2').colors))
tumor=colorFader(general[2], c2='#FFFFFF', n=5, mix=0)
lymphoid=general[4]
myeloid=general[1]
Blike=colorFader(general[0], c2='#FFFFFF', n=2, mix=0)
neutrophil=colorFader(general[3], c2='#FFFFFF', n=2, mix=0)
stroma=general[5]

leiden_col = [tumor[0], tumor[1], tumor[2], tumor[3], tumor[4], neutrophil[0], neutrophil[1], stroma,  Blike[0],  Blike[1], lymphoid, myeloid]

adata.uns['latent_leiden_0.35_colors'] = leiden_col
adata.uns['niche_colors'] = leiden_col

# %%
# Compute integration metric
nc_silouette = silhouette_score(X=adata[adata.obs.patient == 'Lung5'].obsm['nichecompass_latent'], labels=adata[adata.obs.patient == 'Lung5'].obs['mapping_entity'])
print(nc_silouette)

# %%
### Supplementary Fig. 29b ###
adata.uns['batch_colors'] = list(np.array(adata.uns['batch_colors'])[[0,1,6,2,3,4,5]])
sc.pl.umap(adata, color=['batch'], ncols=4, wspace=0.3, size=0.5)
plt.savefig(f"{figure_folder_path}/s29_b.svg", bbox_inches="tight", format='svg')
sc.set_figure_params(fontsize=12, dpi=100, figsize=(4,3))

# %%
# Perform label transfer
# Get query active GPs to subset the reference on
model =  NicheCompass.load(dir_path=f'{base_path}/artifacts/{dataset}/models/{model_labels[1]}/{load_timestamps[1]}/',
                           adata=None,
                           adata_file_name=f'{dataset}_{model_labels[1]}.h5ad',
                           gp_names_key='nichecompass_gp_names')
query_active_gps = model.get_active_gps()

del model
gc.collect()

# %%
# Load reference adata and pre-process labels
adata_ref, adata_path, _, _, _ = load_adata(
    load_timestamp=load_timestamps[0],
    model_label=model_labels[0])

leiden2niche = {
    '0': '1- Tumor (stroma border)', '2': '2- Tumor interior', '5': '3- Tumor (neutrophil border)', '7': '4- Tumor interior', '11': '5- Infiltrated tumor',
    '3': '6- Tumor-infiltrated neutrophils', '4': '7- Neutrophil expansion', '1': '8- Stroma', '6': '9- Plasmablast rich stroma', '9': '10- Lymphoid rich stroma', '8': '11- Lymphoid aggregates', '10': '12- Macrophage rich stroma'
}

adata_ref.obs['niche'] = adata_ref.obs['latent_leiden_0.45'].map(leiden2niche)
adata_ref.obs['niche_0.1'] = adata_ref.obs['latent_leiden_0.1'].map({'0':'lymphoid stroma', '1':'tumor 9,12', '2': 'myeloid stroma', '3': 'tumor 6', '4': 'tumor 5'})
adata_ref.obs['niche_0.1lo'] = adata_ref.obs['latent_leiden_0.1'].map({'0':'lymphoid stroma', '1':'tumor', '2': 'myeloid stroma', '3': 'tumor', '4': 'tumor'})

general=np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('Dark2').colors))
tumor=colorFader(general[2], c2='#FFFFFF', n=5, mix=0)
lymphoid=general[4]
myeloid=general[1]
Blike=colorFader(general[0], c2='#FFFFFF', n=2, mix=0)
neutrophil=colorFader(general[3], c2='#FFFFFF', n=2, mix=0)
stroma=general[5]


leiden_colDict = {
    '0': tumor[0], '2': tumor[1], '5': tumor[2], '7': tumor[3], '11': tumor[4], 
    '3': neutrophil[0], '4': neutrophil[1], 
    '1': stroma, '6': Blike[0], '9': Blike[1], '8': lymphoid, '10': myeloid 
}
adata_ref.uns['latent_leiden_0.45_colors'] = [x for x in leiden_colDict.values()]
adata_ref.uns['leidenOrd_colors'] = [x for x in leiden_colDict.values()]
adata_ref.uns['niche_colors'] = [x for x in leiden_colDict.values()]
colDict = {cl: color for color, cl in zip(adata_ref.uns['niche_colors'], adata_ref.obs.niche.cat.categories)}

# Subset GPs to those active both in query and reference
ref_active_gps = adata_ref.uns['nichecompass_active_gp_names']
gps = list(set(ref_active_gps).intersection(set(query_active_gps)))
adata_ref.obsm['nichecompass_latent'] = adata_ref.obsm['nichecompass_latent'][:,pd.Series(ref_active_gps).isin(gps)]
gc.collect()

# Fit kNN classifier on the reference
knn = KNeighborsClassifier(n_neighbors=50, weights='distance')
knn.fit(X=adata_ref.obsm['nichecompass_latent'], y=adata_ref.obs[['niche','niche_0.1','niche_0.1lo']])

del adata_ref
gc.collect()

# %%
# Load query
adata_query, adata_path, _, _, _ = load_adata(
    load_timestamp=load_timestamps[1],
    model_label=model_labels[1])

adata_query.obs['mapping_entity'].value_counts()
adata_query  = adata_query[adata_query.obs.mapping_entity=='query']
adata_query.obsm['nichecompass_latent'] = adata_query.obsm['nichecompass_latent'][:,pd.Series(query_active_gps).isin(gps)]
gc.collect()

# %%
# Predict labels with the knn classifier
proba = knn.predict_proba(adata_query.obsm['nichecompass_latent'])
k_dist, k_indx = knn.kneighbors(adata_query.obsm['nichecompass_latent'], n_neighbors=50, return_distance=True)

predictions = proba[0]
predictions = pd.DataFrame({'predlabel': np.argmax(predictions, axis=1), 'probability': np.max(predictions, axis=1), 'mean_dist': np.mean(k_dist, axis=1), 'k_dist': k_dist[:,49]})
predictions['predlabel'] = predictions['predlabel'].map({i: l for i, l in enumerate(knn.classes_[0])})
predictions.index = adata_query.obs.index

predictions.to_csv(f'{result_folder_path}/label_transfer.csv')

# %%
# Load labels predicted with knn classifier
predictions = pd.read_csv(f'{result_folder_path}/label_transfer.csv', index_col=0)
predictions['predlabel'] = predictions['predlabel'].apply(lambda x: x.replace('6- Neutrophil expansion', '6- Tumor-infiltrated neutrophils').replace('7- Myeloid cells rich stroma', '7- Neutrophil expansion'))
adata_query.obs = pd.concat([adata_query.obs, predictions], axis=1)
adata_query.obs.predlabel = pd.Categorical(adata_query.obs.predlabel, categories=colDict.keys())
adata_query.uns['predlabel_colors'] = [colDict[cl] for cl in adata_query.obs.predlabel.cat.categories]

# %%
### Supplementary Fig. 29c,d,e ###
# plot only most present niches (>5%)
keep = predictions['predlabel'].value_counts().index[predictions['predlabel'].value_counts()/predictions.shape[0] > 0.05]
adata_query = adata_query[adata_query.obs.predlabel.isin(keep)]

sc.set_figure_params(dpi=300, figsize=(6,4))
sc.pl.umap(adata_query,
           color=['cell type'],
           size=2,
           show=False,
           frameon=False,
           ncols=1,
           wspace=3)
plt.savefig(f"{figure_folder_path}/s29_c.svg", bbox_inches="tight", format='svg')
sc.pl.umap(adata_query,
           color=['predlabel'],
           size=1.5,
           show=False,
           frameon=False,
           ncols=1,
           wspace=3)
plt.savefig(f"{figure_folder_path}/s29_d.svg", bbox_inches="tight", format='svg')
sc.pl.umap(adata_query,
           color=['probability'],
           size=1.5,
           color_map='viridis',
           show=False,
           frameon=False,
           ncols=1,
           wspace=3,
           vmax=1,
           vmin=0)
plt.savefig(f"{figure_folder_path}/s29_e.svg", bbox_inches="tight", format='svg')
sc.set_figure_params(dpi=300, figsize=(4,3))

# %%
### Supplementary Fig. 30a ###
print((predictions.probability < 0.7).value_counts()/(74045+17646))
predictions.probability.hist()
plt.savefig(f"{figure_folder_path}/s30_a1.svg", bbox_inches="tight", format='svg')

# %%
adata, \
adata_path, \
model_folder_path, \
figure_folder_path, \
result_folder_path = load_adata(load_timestamp=load_timestamps[2],
                                model_label=model_labels[2])

# %%
plot_latent(adata, model='reference_query_mapping')

# %%
latent_leiden_resolution=0.7
res_details(adata, resolution=latent_leiden_resolution, model='reference_query_mapping')

# %%
### Fig. 5f ###
sc.set_figure_params(dpi=300, figsize=(4,3))
sc.pl.umap(adata[adata.obs.mapping_entity=='reference'], color=['mapping_entity'], ncols=1, wspace=0.3, size=0.5,frameon=False, show=False)
plt.savefig(f"{figure_folder_path}/5_f1.svg", bbox_inches="tight", format='svg')
sc.pl.umap(adata, color=['mapping_entity'], ncols=1, wspace=0.3, size=0.5,frameon=False, show=False)
plt.savefig(f"{figure_folder_path}/5_f2.svg", bbox_inches="tight", format='svg')
sc.set_figure_params(fontsize=12, dpi=300, figsize=(4,3))

# %%
latent_leiden_resolution = 0.7
sc.tl.dendrogram(adata, groupby=f"latent_leiden_{latent_leiden_resolution}", use_rep='nichecompass_latent', n_pcs=adata.obsm['nichecompass_latent'].shape[1])
adata.obs[f'latent_leiden_{latent_leiden_resolution}'] = adata.obs[f'latent_leiden_{latent_leiden_resolution}'].cat.reorder_categories(adata.uns[f'dendrogram_latent_leiden_{latent_leiden_resolution}']['categories_ordered'], ordered=False)

rcParams['figure.figsize'] = (8, 6)
sc.pl.umap(adata, color=['latent_leiden_0.7'], size=2.5, legend_loc='on data', frameon=False)
rcParams['figure.figsize'] = (4, 3)

# %%
leiden2niche = {
    '0': '1- Tumor (stroma border)', '8': '1- Tumor (stroma border)', '1': '2- Tumor interior', '5': '3- Tumor (neutrophil border)', '7': '4- Tumor interior', '19': '4- Tumor interior', 
    '17': '5- Infiltrated tumor', '12': '6- Tumor-infiltrating neutrophils', '6': '7- Neutrophil expansion', '13': '8- Stroma', '14': '8- Stroma', '4': '8- Stroma', '15': '8- Stroma', '11': '9- Plasmablast rich stroma', 
    '10': '10- Lymphoid rich stroma', '2': '11- Lymphoid aggregates', 
    '9': '15- Tumor (macrophage infiltrated)', '18': '13- Infiltrating macrophages', '16': '12- Macrophage rich stroma', '3': '14- Immune rich stroma'
}

adata.obs['niche'] = adata.obs['latent_leiden_0.7'].map(leiden2niche)
adata.obs['leidenOrd'] = adata.obs['niche'].apply(lambda x: x.split('-')[0])
sc.pl.umap(adata, color=['niche','leidenOrd'], size=2.5, legend_loc='on data', frameon=False)

rcParams['figure.figsize'] = (4, 0.5)
general=np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('Dark2').colors))
a=np.outer(np.ones(len(general)),np.arange(0,1,0.01))   # pseudo image data
plt.imshow(a,aspect='auto',cmap=plt.get_cmap('Dark2'),origin="lower")

general=np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('Dark2').colors))
tumor=colorFader(general[2], c2='#FFFFFF', n=6, mix=0)
lymphoid=general[4]
myeloid=colorFader(general[1], c2='#FFFFFF', n=2, mix=0)
Blike=colorFader(general[0], c2='#FFFFFF', n=2, mix=0)
neutrophil=colorFader(general[3], c2='#FFFFFF', n=2, mix=0)
stroma=colorFader(general[5], c2='#FFFFFF', n=2, mix=0)


leiden_colDict = {
    '1': tumor[0], '2': tumor[1], '3': tumor[2], '4': tumor[3], '5': tumor[4],
    '7': neutrophil[0], '6': neutrophil[1], 
    '8': stroma[0], '10': Blike[0], '9': Blike[1], '11': lymphoid, '12': myeloid[0],
    '13': myeloid[1], '14': stroma[1], '15': tumor[5]
}
# adata.uns['latent_leiden_0.7_colors'] = [x for x in leiden_colDict.values()]
adata.uns['leidenOrd_colors'] = [x for x in leiden_colDict.values()]
adata.uns['niche_colors'] = [x for x in leiden_colDict.values()]

# %%
### Fig. 5i ###
rcParams['figure.figsize'] = (8, 6)
sc.pl.umap(adata, color=['leidenOrd'], size=2.5, legend_loc='on data', frameon=False, show=False)
plt.savefig(f"{figure_folder_path}/5_i1.svg", bbox_inches="tight", format='svg')

sc.pl.umap(adata, color=['niche'], size=2.5, frameon=False, show=False)
plt.savefig(f"{figure_folder_path}/5_i2.svg", bbox_inches="tight", format='svg')
rcParams['figure.figsize'] = (4, 3)

cluster_var = 'batch'
plot_var = 'leidenOrd'
props = adata.obs.groupby([cluster_var, plot_var]).size().reset_index()
props = props.pivot(columns=plot_var, index=cluster_var).T
props.index = props.index.droplevel(0)
props.fillna(0, inplace=True)
props = props.div(props.sum(axis=1), axis=0)*100 
plot_cluster_proportions(props, xlabel_rotation=90, cluster_palette=adata.uns[f'{cluster_var}_colors'], figsize=(8,3), remove_legend=False)
fig = plot_cluster_proportions(props.loc[props.lung13>5,:], xlabel_rotation=90, cluster_palette=adata.uns[f'{cluster_var}_colors'], figsize=(2.8,3), remove_legend=False)
plt.savefig(f"{figure_folder_path}/5_i3.svg", bbox_inches="tight", format='svg')

# %%
### Supplementary figure 30b ###
cluster_var = 'cell type'
plot_var = 'leidenOrd'

props = adata[(adata.obs.batch=='lung13') & adata.obs.leidenOrd.isin(['10','14'])].obs.groupby([cluster_var, plot_var]).size().reset_index()
props = props.pivot(columns=plot_var, index=cluster_var).T
props.index = props.index.droplevel(0)
props.fillna(0, inplace=True)
props = props.div(props.sum(axis=1), axis=0)*100 
fig = plot_cluster_proportions(props, xlabel_rotation=90, cluster_palette=adata.uns[f'{cluster_var}_colors'], figsize=(2,3))
plt.savefig(f"{figure_folder_path}/s30_b1.svg", bbox_inches="tight", format='svg')

props = adata[(adata.obs.batch.isin(['lung9_rep1','lung9_rep2'])) & adata.obs.leidenOrd.isin(['10','14'])].obs.groupby([cluster_var, plot_var]).size().reset_index()
props = props.pivot(columns=plot_var, index=cluster_var).T
props.index = props.index.droplevel(0)
props.fillna(0, inplace=True)
props = props.div(props.sum(axis=1), axis=0)*100 
fig = plot_cluster_proportions(props, xlabel_rotation=90, cluster_palette=adata.uns[f'{cluster_var}_colors'], figsize=(2,3))
plt.savefig(f"{figure_folder_path}/s30_b2.svg", bbox_inches="tight", format='svg')

# %%
### Fig. 5j ###
n = 3
fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(8*n,6*2))

# for i, batch in enumerate(adata.obs.batch.unique()):
for i, batch in enumerate(['lung13', 'lung6', 'lung9_rep1']):
    ax = sc.pl.embedding(adata[adata.obs.batch==batch], basis="spatial", color=['cell type'], size=6, legend_loc=None, frameon=False, title=[''], ax=axes[0,i], show=False)
    ax = sc.pl.embedding(adata[adata.obs.batch==batch], basis="spatial", color=['leidenOrd'], size=6, legend_loc=None, frameon=False, title=[''], ax=axes[1,i], show=False)

fig.tight_layout()
fig.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig(f"{figure_folder_path}/5_j.svg", bbox_inches="tight", format='svg')

# %%
### Fig. 5k ###
leiden_col_key='leidenOrd'
n = 25
structure = 'tumor_clusters'
groups = ['1', '2', '3', '4', '5', '15']

# compute n neighbours per cell type
knn = {}
cell_counts = {}
for b in adata.obs.batch.unique():
    X = adata[adata.obs.batch==b].obsm['spatial']
    celltypes = adata[adata.obs.batch==b].obs['cell type'].astype(str).values.astype('<U22')    
    cellnames = adata[adata.obs.batch==b].obs_names

    knn[b] = NearestNeighbors(n_neighbors=n)
    knn[b].fit(X)
    knn[b] = knn[b].kneighbors(X, return_distance=False)    
    knn[b] = celltypes[knn[b]]

    for i in range(len(cellnames)):
        unique, counts = np.unique(knn[b][i,:], return_counts=True)
        cell_counts[cellnames[i]] = dict(zip(unique, counts))

adata.obsm[f'k{n}_neighbours_celltype'] = pd.DataFrame(cell_counts).T.fillna(0)

# format for plotting and keep only those with an interesting number
dt = adata.obsm[f'k{n}_neighbours_celltype']
dt[leiden_col_key] = adata.obs[leiden_col_key]
dt = dt.loc[dt[leiden_col_key].isin(groups),:]
dt[leiden_col_key] = dt[leiden_col_key].cat.remove_unused_categories()
dt = dt.reset_index()
dt = pd.melt(dt, id_vars=['index',leiden_col_key], value_name=f'n NN/{n}', var_name='cell type')
dt['cell type'] = pd.Categorical(dt['cell type'], categories=adata.obs['cell type'].cat.categories)
include = (dt.groupby(['cell type',leiden_col_key])[f'n NN/{n}'].mean().groupby(['cell type']).max() > n*0.05) & (dt.groupby(['cell type',leiden_col_key])[f'n NN/{n}'].mean().groupby(['cell type']).max() < n*0.6)
dt = dt[dt['cell type'].isin(include[include].index.values)]
dt['cell type'] = dt['cell type'].cat.remove_unused_categories()

# %%
### Fig. 5k ###
rcParams['figure.figsize'] = (len(groups)*1.5+0.3, 4)
sns.boxplot(data=dt, x=leiden_col_key, y=f'n NN/{n}', hue='cell type', palette=adata.uns[f'cell type_colors'][include.values], showfliers = False).set(title='')
plt.ylim((0, 25))
plt.xticks(rotation=90)
plt.xlabel("Tumor niches", fontsize=17.5, rotation=180)
plt.ylabel(f'Number of cells in\n 25 closest neighbors', fontsize=17.5, labelpad=20)
plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1.2, 0.7), fontsize=15)
plt.tick_params(bottom=False, labelsize=17.5)
sns.despine(offset=10, trim=True, bottom=True, right=False)
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.gca().spines["left"].set_visible(False)
plt.yticks(rotation=90)
plt.savefig(f"{figure_folder_path}/5_k.svg", bbox_inches="tight", format='svg')
plt.show()
rcParams['figure.figsize'] = (4, 3)

# %%
# Get number of cells for figure caption
dt[['index','leidenOrd']].drop_duplicates()['leidenOrd'].value_counts()

# %%
### Fig. 5l ###
interest_gps = ['SPP1_ligand_receptor_GP', 'Spp1_ligand_receptor_target_gene_GP']
invert = ['SPP1_ligand_receptor_GP']

# Create active gene program df
interest_gp_df = pd.DataFrame(adata.obsm['nichecompass_latent'][:,pd.Series(adata.uns['nichecompass_active_gp_names']).isin(interest_gps).values],
                            columns=adata.uns['nichecompass_active_gp_names'][pd.Series(adata.uns['nichecompass_active_gp_names']).isin(interest_gps).values])
interest_gp_df = interest_gp_df.set_index(adata.obs.index)

# Drop columns if they are already in ´adata.obs´ and invert if apropriate
for gp in interest_gps:
    if gp in adata.obs:
        adata.obs.drop(gp, axis=1, inplace=True)
        
    if gp in invert:
        interest_gp_df[gp] = -interest_gp_df[gp]

# Concatenate active gene program df horizontally to ´adata.obs´
adata.obs = pd.concat([adata.obs, interest_gp_df], axis=1)

sc.set_figure_params(dpi=300, figsize=(4,3))

for gp in interest_gps:
    gp_idx = adata.uns['nichecompass_gp_names'].tolist().index(gp)
    sc.pl.umap(adata, color=[gp], ncols=2, size=0.5, frameon=False, show=False, cmap='RdGy_r')
    plt.savefig(f"{figure_folder_path}/5_l{gp}.svg", bbox_inches="tight", format='svg')

sc.pl.umap(adata, color=['SPP1','EGFR','ITGAV'], ncols=3, size=0.5, frameon=False, show=False, cmap='RdPu')
plt.savefig(f"{figure_folder_path}/5_l.svg", bbox_inches="tight", format='svg')

# %%
### Supplementary Fig. 31 ###
# Cell types markers
markers = ["CD68", "MARCO", "CD14", "SPP1", "CXCL9", "IFI27", "CD9", "FN1", "TIMP1", "COL3A1", "COL1A1", "MMP12", "MMP2"]

adata_tmp = adata[(adata.obs['cell type']=='macrophage')]
adata_tmp.obs['batch'] = adata_tmp.obs['batch'].str.replace('lung','Lung ').str.replace('_rep',' Replicate ')

rcParams['figure.figsize'] = (6, 4)
keep = adata.obs['niche'].value_counts().index[adata.obs['niche'].value_counts() > 1000]
adata_tmp = adata_tmp[adata_tmp.obs['niche'].isin(keep)]

rcParams['figure.figsize'] = (6, 4)
sc.pl.umap(adata_tmp,
           color=['cell type', 'batch', 'niche'],
           ncols=3,
           size=0.5,
           frameon=False,
           show=False,
           wspace=0.25)
plt.savefig(f"{figure_folder_path}/s31_a.svg", bbox_inches="tight", format='svg')

sc.pl.umap(adata_tmp,
           color=['MARCO', 'SPP1', 'IFI27', 'CD9', 'FN1', 'TIMP1', 'COL3A1', 'COL1A1', 'MMP12', 'MMP2'],
           size=0.5,
           frameon=False,
           show=False,
           cmap='RdPu',
           ncols=4,
           colorbar_loc=None)
plt.savefig(f"{figure_folder_path}/s31_b.svg", bbox_inches="tight", format='svg')

sc.pl.dotplot(adata_tmp, markers, groupby='batch', dendrogram=False, show=False, cmap="Reds")
plt.savefig(f"{figure_folder_path}/s31_c.svg", bbox_inches="tight", format='svg')

# %%
# Load query to select shared GPs
model =  NicheCompass.load(dir_path=f'{base_path}/artifacts/{dataset}/models/{model_labels[2]}/{load_timestamps[2]}/',
                  adata=None,
                  adata_file_name=f'{dataset}_{model_labels[2]}.h5ad',
                  gp_names_key='nichecompass_gp_names')
query_active_gps = model.get_active_gps()

# Load reference adata and preprocess
adata_ref, adata_path, _, _, _ = load_adata(
    load_timestamp=load_timestamps[0],
    model_label=model_labels[0])

leiden2niche = {
    '0': '1- Tumor (stroma border)', '2': '2- Tumor interior', '5': '3- Tumor (neutrophil border)', '7': '4- Tumor interior', '11': '5- Infiltrated tumor',
    '3': '6- Tumor-infiltrating neutrophils', '4': '7- Neutrophil expansion', '1': '8- Stroma', '6': '9- Plasmablast rich stroma', '9': '10- Plasmablast rich stroma', '8': '11- Lymphoid aggregates', '10': '12- Macrophage rich stroma'
}

adata_ref.obs['niche'] = adata_ref.obs['latent_leiden_0.45'].map(leiden2niche)
adata_ref.obs['niche_0.1'] = adata_ref.obs['latent_leiden_0.1'].map({'0':'lymphoid stroma', '1':'tumor 9,12', '2': 'myeloid stroma', '3': 'tumor 6', '4': 'tumor 5'})
adata_ref.obs['niche_0.1lo'] = adata_ref.obs['latent_leiden_0.1'].map({'0':'lymphoid stroma', '1':'tumor', '2': 'myeloid stroma', '3': 'tumor', '4': 'tumor'})

general=np.apply_along_axis(to_hex, 1, np.array(plt.get_cmap('Dark2').colors))
tumor=colorFader(general[2], c2='#FFFFFF', n=5, mix=0)
lymphoid=general[4]
myeloid=general[1]
Blike=colorFader(general[0], c2='#FFFFFF', n=2, mix=0)
neutrophil=colorFader(general[3], c2='#FFFFFF', n=2, mix=0)
stroma=general[5]

leiden_colDict = {
    '0': tumor[0], '2': tumor[1], '5': tumor[2], '7': tumor[3], '11': tumor[4], 
    '3': neutrophil[0], '4': neutrophil[1], 
    '1': stroma, '6': Blike[0], '9': Blike[1], '8': lymphoid, '10': myeloid 
}
adata_ref.uns['latent_leiden_0.45_colors'] = [x for x in leiden_colDict.values()]
adata_ref.uns['leidenOrd_colors'] = [x for x in leiden_colDict.values()]
adata_ref.uns['niche_colors'] = [x for x in leiden_colDict.values()]

colDict = {cl: color for color, cl in zip(adata_ref.uns['niche_colors'], adata_ref.obs.niche.cat.categories)}
ref_active_gps = adata_ref.uns['nichecompass_active_gp_names']
gps = list(set(ref_active_gps).intersection(set(query_active_gps)))
adata_ref.obsm['nichecompass_latent'] = adata_ref.obsm['nichecompass_latent'][:,pd.Series(ref_active_gps).isin(gps)]
gc.collect()

# fit knn on reference
knn = KNeighborsClassifier(n_neighbors=50, weights='distance')
knn.fit(X=adata_ref.obsm['nichecompass_latent'], y=adata_ref.obs[['niche','niche_0.1','niche_0.1lo']])
del adata_ref
gc.collect()

# load query
adata_query, adata_path, _, _, _ = load_adata(
    load_timestamp=load_timestamps[2],
    model_label=model_labels[2])
adata_query.obs['mapping_entity'].value_counts()
adata_query  = adata_query[adata_query.obs.mapping_entity=='query']
adata_query.obsm['nichecompass_latent'] = adata_query.obsm['nichecompass_latent'][:,pd.Series(query_active_gps).isin(gps)]
gc.collect()

# %%
### Supplementary Fig. 30a ###
predictions = pd.read_csv(f'{result_folder_path}/label_transfer.csv', index_col=0)
sc.set_figure_params(dpi=300, figsize=(4,3.3))
print((predictions.probability < 0.7).value_counts()/predictions.probability.shape[0])
print((predictions.probability < 0.5).value_counts()/predictions.probability.shape[0])
predictions.probability.hist()
plt.savefig(f"{figure_folder_path}/s30_a2.svg", bbox_inches="tight", format='svg')

# %%
### Fig. 5g-h ###
# Simplify labels by keeping only niches that are assigned to at least 5% of the cells
keep = predictions['predlabel'].value_counts().index[predictions['predlabel'].value_counts()/predictions.shape[0] > 0.05]
adata_query.obs = pd.concat([adata_query.obs, predictions], axis=1)
adata_query.obs.predlabel = pd.Categorical(adata_query.obs.predlabel, categories=colDict.keys())
adata_query.uns['predlabel_colors'] = [colDict[cl] for cl in adata_query.obs.predlabel.cat.categories]

sc.set_figure_params(dpi=300, figsize=(6,4))
sc.pl.umap(adata_query, color=['cell type'], size=0.5, frameon=False, show=False, ncols=1, wspace=3, title=[''])
plt.savefig(f"{figure_folder_path}/5_g.svg", bbox_inches="tight", format='svg')

sc.pl.umap(adata_query, color=['predlabel'], size=0.5, frameon=False, show=False, ncols=1, wspace=3, title=[''])
plt.savefig(f"{figure_folder_path}/5_h1.svg", bbox_inches="tight", format='svg')

sc.pl.umap(adata_query, color=['probability'], size=0.5, frameon=False, show=False, ncols=1, wspace=3, vmax=1, vmin=0, colorbar_loc='bottom', title=[''], color_map='viridis')
plt.savefig(f"{figure_folder_path}/5_h2.svg", bbox_inches="tight", format='svg')

sc.set_figure_params(dpi=300, figsize=(4,3))

# %%
# Load communication network results
nx_s6 = pd.read_csv(f'{result_folder_path}/Spp1_ligand_receptor_target_gene_GP_lung6.csv', index_col=0)
nx_s13 = pd.read_csv(f'{result_folder_path}/Spp1_ligand_receptor_target_gene_GP_lung13.csv', index_col=0)

# %%
# Rename niches from the latent_leiden_0.7 to the ordered leidenOrd labels we use throughout this section
niche2niche = adata.obs[['latent_leiden_0.7','leidenOrd']].drop_duplicates().set_index('latent_leiden_0.7').to_dict()['leidenOrd']
nx_s6['source'] = nx_s6['source'].astype(str).map(niche2niche)
nx_s6['target'] = nx_s6['target'].astype(str).map(niche2niche)
nx_s13['source'] = nx_s13['source'].astype(str).map(niche2niche)
nx_s13['target'] = nx_s13['target'].astype(str).map(niche2niche)

# %%
# Re-scale
min_value = min(nx_s6["strength_unscaled"].min(), nx_s13["strength_unscaled"].min())
max_value = max(nx_s6["strength_unscaled"].max(), nx_s13["strength_unscaled"].max())
nx_s13["strength"] = (nx_s13["strength_unscaled"] - min_value) / (max_value - min_value)
nx_s13["strength"] = np.round(nx_s13["strength"], 2)
nx_s6["strength"] = (nx_s6["strength_unscaled"] - min_value) / (max_value - min_value)
nx_s6["strength"] = np.round(nx_s6["strength"], 2)
nx_s6 = nx_s6[nx_s6.strength>0]
nx_s13 = nx_s13[nx_s13.strength>0]

# %%
### Fig 5m1 ###
nx = nx_s13.copy()
niches = nx.source.unique().tolist() + nx.target.unique().tolist()
base = alt.Chart(nx[nx.strength > 0]).mark_point(
    filled=True,
    size=2000,
    shape='square',
    opacity=0.6,
    strokeWidth=0
).encode(
    x=alt.X('target:O', title=None, axis=alt.Axis(orient='bottom', labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=niches)),    
    y=alt.Y('source:O', title=None, axis=alt.Axis(labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=niches)),
    color=alt.Color('strength:Q', scale=alt.Scale(scheme='yellowgreenblue', domain=[0,1]))
).properties(
    width=len(niches)*27,
    height=27*len(niches)
)

text = base.mark_text().encode(
    text='strength:Q',
    color=alt.value("black")
)
chart = base + text
chart.save(f"{figure_folder_path}/5_m1.svg")
chart

# %%
### Fig 5m2 ###
nx = nx_s6.copy()
niches = nx.source.unique().tolist() + nx.target.unique().tolist()
base = alt.Chart(nx[nx.strength > 0]).mark_point(
    filled=True,
    size=2000,
    shape='square',
    opacity=0.6,
    strokeWidth=0
).encode(
    x=alt.X('target:O', title=None, axis=alt.Axis(orient='bottom', labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=niches)),    
    y=alt.Y('source:O', title=None, axis=alt.Axis(labelFontSize=15, titleFontSize=15), scale=alt.Scale(domain=niches)),
    color=alt.Color('strength:Q', scale=alt.Scale(scheme='yellowgreenblue', domain=[0,1]))
).properties(
    width=len(niches)*27,
    height=27*len(niches)
)

text = base.mark_text().encode(
    text='strength:Q',
    color=alt.value("black")
)
chart = base + text
chart.save(f"{figure_folder_path}/5_m2.svg")
chart

# %%

