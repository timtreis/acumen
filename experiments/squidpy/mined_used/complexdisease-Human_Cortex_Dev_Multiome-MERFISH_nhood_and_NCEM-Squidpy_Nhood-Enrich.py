# mined from: https://github.com/complexdisease/Human_Cortex_Dev_Multiome/blob/0cf3257c071f64dc1c8fc6cbc9bccb25ba3a9cc5/MERFISH_nhood_and_NCEM/Squidpy_Nhood-Enrich.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment, squidpy.read.vizgen

# %%
import ncem as nc
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import pandas as pd 

sc.settings.set_figure_params(dpi=80)
import warnings
warnings.filterwarnings("ignore")

sc.logging.print_header()

print(f"ncem=={nc.__version__}")

from ncem.interpretation import InterpreterInteraction
from ncem.data import get_data_custom, customLoader

# %%
my_colors = {'RG-tRG': '#F6222E', 'RG-oRG': '#FE00FA', 'IPC-EN': '#16FF32', 'EN-Newborn': '#3283FE', 'EN-IT-Immature': '#FEAF16', 'EN-L2_3-IT': '#B00068', 'EN-L4-IT': '#1CFFCE', 'EN-L5-IT': '#90AD1C', 'EN-L6-IT': '#2ED9FF', 'EN-L5-ET': '#AA0DFE', 'EN-L5_6-NP': '#F8A19F', 'EN-L6-CT': '#325A9B', 'EN-L6b': '#C4451C', 'IN-dLGE-Immature': '#1C8356', 'IN-CGE-Immature': '#85660D', 'IN-CGE-VIP': '#B10DA1', 'IN-CGE-SNCG': '#FBE426', 'IN-CGE-LAMP5': '#1CBE4F', 'IN-MGE-Immature': '#FA0087', 'IN-MGE-SST': '#FC1CBF', 'IN-MGE-PV': '#F7E1A0', 'IPC-Glia': '#C075A6', 'Astrocyte-Protoplasmic': '#AAF400', 'Astrocyte-Fibrous': '#BDCDFF', 'OPC': '#B5EFB5', 'Oligodendrocyte-Immature': '#7ED7D1', 'Oligodendrocyte': '#1C7F93', 'Microglia': '#683B79', 'Vascular': '#3B00FB'}

# %%
ad = sq.read.vizgen(path='./',
    counts_file="./ARKFrozen65V1/cell_by_gene.csv",
    meta_file="./ARKFrozen65V1/cell_metadata.csv")

annot = pd.read_csv('./ARKFrozen65V1/ARKFrozen65V1_annotation.csv',
                   sep=',', index_col=0)

annot = annot[['type']]

ad = ad[annot.index.astype(str),]

annot.index = annot.index.astype(str)

ad.obs = ad.obs.join(annot)

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, inplace=True)
sc.pp.log1p(ad)
sc.pp.pca(ad)
sc.pp.neighbors(ad)

ad.obs['Cluster'] = ad.obs['type']

# %%
sc.pl.embedding(ad, basis="spatial", color='type', palette={k: v for k, v in my_colors.items() if k in ad.obs.type.value_counts().index})

# %%
sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key="spatial")
sq.gr.nhood_enrichment(ad, cluster_key="type")

# %%
sq.pl.nhood_enrichment(
    ad,
    cluster_key="type",
    method=None,
    cmap="inferno",
    vmin=-50,
    vmax=100,figsize=(8, 8))

# %%
old_order = sorted(list(my_colors.keys()))

zscore_data = ad.uns['type_nhood_enrichment']['zscore']
zscore_ordered = pd.DataFrame(zscore_data, columns=old_order, index=old_order)

zscore_reordered = zscore_ordered.reindex(index=list(my_colors.keys()), columns=list(my_colors.keys()))

# %%
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(
    zscore_reordered.values,
    cmap='inferno',
    vmin=-50,
    vmax=100
)

ax.set_title('Neighborhood enrichment')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)

n_clusters = len(list(my_colors.keys()))
ax.set_xticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_yticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_xticklabels(list(my_colors.keys()), rotation=90)
ax.set_yticklabels(list(my_colors.keys()))

cbar = plt.colorbar(heatmap, shrink=0.7, aspect=30)
cbar.set_label('')

plt.tight_layout()

plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/ARKFrozen65V1/ARKFrozen65V1_wholedataset_nhood_enrich.svg', format='svg')
plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/ARKFrozen65V1/ARKFrozen65V1_wholedataset_nhood_enrich.pdf', format='pdf')

plt.show()

# %%
ad = sq.read.vizgen(path='./',
    counts_file="./ARKFrozen65V1/subset_cell_by_gene.csv",
    meta_file="./ARKFrozen65V1/subset_cell_metadata.csv")

annot = pd.read_csv('./ARKFrozen65V1/ARKFrozen65V1_annotation.csv',
                   sep=',', index_col=0)

annot = annot[['type']]

ad = ad[annot.index.astype(str),]

annot.index = annot.index.astype(str)

ad.obs = ad.obs.join(annot)

#Downsample
print('downsampling to 60k')
ad = ad[ad.obs.sample(n=60000, random_state=1).index]

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, inplace=True)
sc.pp.log1p(ad)
sc.pp.pca(ad)
sc.pp.neighbors(ad)

ad.obs['Cluster'] = ad.obs['type']

sc.pl.embedding(ad, basis="spatial", color='type', palette={k: v for k, v in my_colors.items() if k in ad.obs.type.value_counts().index})

# %%
sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key="spatial", radius=50)
sq.gr.nhood_enrichment(ad, cluster_key="type")

# %%
sq.pl.nhood_enrichment(
    ad,
    cluster_key="type",
    method=None,
    cmap="inferno",
    vmin=-50,
    vmax=100,
    figsize=(8, 8))

# %%
old_order = sorted(list(my_colors.keys()))

zscore_data = ad.uns['type_nhood_enrichment']['zscore']
zscore_ordered = pd.DataFrame(zscore_data, columns=old_order, index=old_order)

zscore_reordered = zscore_ordered.reindex(index=list(my_colors.keys()), columns=list(my_colors.keys()))

# %%
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(
    zscore_reordered.values,
    cmap='inferno',
    vmin=-50,
    vmax=100
)

ax.set_title('Neighborhood enrichment')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)

n_clusters = len(list(my_colors.keys()))
ax.set_xticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_yticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_xticklabels(list(my_colors.keys()), rotation=90)
ax.set_yticklabels(list(my_colors.keys()))

cbar = plt.colorbar(heatmap, shrink=0.7, aspect=30)
cbar.set_label('')

plt.tight_layout()

plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/ARKFrozen65V1/ARKFrozen65V1_60kcells_nhood_enrich.svg', format='svg')
plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/ARKFrozen65V1/ARKFrozen65V1_60kcells_nhood_enrich.pdf', format='pdf')

plt.show()

# %%
ad = sq.read.vizgen(path='./',
    counts_file="./ARKFrozen62PFC/cell_by_gene.csv",
    meta_file="./ARKFrozen62PFC/cell_metadata.csv")

annot = pd.read_csv('./ARKFrozen62PFC/ARKFrozen62PFC_annotation.csv',
                   sep=',', index_col=0)

annot = annot[['type']]

ad = ad[annot.index,]

ad.obs = ad.obs.join(annot)

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, inplace=True)
sc.pp.log1p(ad)
sc.pp.pca(ad)
sc.pp.neighbors(ad)

ad.obs['Cluster'] = ad.obs['type']

# %%
sc.pl.embedding(ad, basis="spatial", color='type', palette={k: v for k, v in my_colors.items() if k in ad.obs.type.value_counts().index})

# %%
sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key="spatial", radius=40)
sq.gr.nhood_enrichment(ad, cluster_key="type")

# %%
sq.pl.nhood_enrichment(
    ad,
    cluster_key="type",
    method=None,
    cmap="inferno",
    vmin=-50,
    vmax=100,
    figsize=(8, 8))

# %%
old_order = sorted(list(my_colors.keys()))

zscore_data = ad.uns['type_nhood_enrichment']['zscore']
zscore_ordered = pd.DataFrame(zscore_data, columns=old_order, index=old_order)

zscore_reordered = zscore_ordered.reindex(index=list(my_colors.keys()), columns=list(my_colors.keys()))

# %%
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(
    zscore_reordered.values,
    cmap='inferno',
    vmin=-50,
    vmax=100
)

ax.set_title('Neighborhood enrichment')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)

n_clusters = len(list(my_colors.keys()))
ax.set_xticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_yticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_xticklabels(list(my_colors.keys()), rotation=90)
ax.set_yticklabels(list(my_colors.keys()))

cbar = plt.colorbar(heatmap, shrink=0.7, aspect=30)
cbar.set_label('')

plt.tight_layout()

plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/ARKFrozen62PFC/ARKFrozen62PFC_nhood_enrich.svg', format='svg')
plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/ARKFrozen62PFC/ARKFrozen62PFC_nhood_enrich.pdf', format='pdf')

plt.show()

# %%
ad = sq.read.vizgen(path='./',
    counts_file="./NIH4365BA10/cell_by_gene.csv",
    meta_file="./NIH4365BA10/cell_metadata.csv")

annot = pd.read_csv('./NIH4365BA10/NIH4365BA10_annotation.csv',
                   sep=',', index_col=0)

annot = annot[['type']]

ad = ad[annot.index,]

ad.obs = ad.obs.join(annot)

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, inplace=True)
sc.pp.log1p(ad)
sc.pp.pca(ad)
sc.pp.neighbors(ad)

ad.obs['Cluster'] = ad.obs['type']

# %%
ad

# %%
sc.pl.embedding(ad, basis="spatial", color='type', palette={k: v for k, v in my_colors.items() if k in ad.obs.type.value_counts().index})

# %%
sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key="spatial", radius=55)
sq.gr.nhood_enrichment(ad, cluster_key="type")

# %%
sq.pl.nhood_enrichment(
    ad,
    cluster_key="type",
    method=None,
    cmap="inferno",
    vmin=-50,
    vmax=100,
    figsize=(8, 8))

# %%


# %%
old_order = sorted(list(my_colors.keys()))

zscore_data = ad.uns['type_nhood_enrichment']['zscore']
zscore_ordered = pd.DataFrame(zscore_data, columns=old_order, index=old_order)

zscore_reordered = zscore_ordered.reindex(index=list(my_colors.keys()), columns=list(my_colors.keys()))

# %%
zscore_reordered['IPC-EN'].iloc[2] = 0 #to remove nan values

# %%
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(
    zscore_reordered.values,
    cmap='inferno',
    vmin=-50,
    vmax=100
)

ax.set_title('Neighborhood enrichment')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)

n_clusters = len(list(my_colors.keys()))
ax.set_xticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_yticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_xticklabels(list(my_colors.keys()), rotation=90)
ax.set_yticklabels(list(my_colors.keys()))

cbar = plt.colorbar(heatmap, shrink=0.7, aspect=30)
cbar.set_label('')

plt.tight_layout()

plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/NIH4365BA10/NIH4365BA10_nhood_enrich.svg', format='svg')
plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/NIH4365BA10/NIH4365BA10_nhood_enrich.pdf', format='pdf')

plt.show()

# %%
ad = sq.read.vizgen(path='./',
    counts_file="./NIH5900BA17/cell_by_gene.csv",
    meta_file="./NIH5900BA17/cell_metadata.csv")

annot = pd.read_csv('./NIH5900BA17/NIH5900BA17_annotation.csv',
                   sep=',', index_col=0)

annot = annot[['type']]

ad = ad[annot.index.astype(str),]

annot.index = annot.index.astype(str)

ad.obs = ad.obs.join(annot)

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, inplace=True)
sc.pp.log1p(ad)
sc.pp.pca(ad)
sc.pp.neighbors(ad)

ad.obs['Cluster'] = ad.obs['type']

# %%
sc.pl.embedding(ad, basis="spatial", color='type', palette={k: v for k, v in my_colors.items() if k in ad.obs.type.value_counts().index})

# %%
sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key="spatial", radius=35)
sq.gr.nhood_enrichment(ad, cluster_key="type", seed=33) #IPC-EN sometimes give nan

# %%
sq.pl.nhood_enrichment(
    ad,
    cluster_key="type",
    method=None,
    cmap="inferno",
    vmin=-50,
    vmax=100,
    figsize=(8, 8))

# %%
old_order = sorted(list(my_colors.keys()))

zscore_data = ad.uns['type_nhood_enrichment']['zscore']
zscore_ordered = pd.DataFrame(zscore_data, columns=old_order, index=old_order)

zscore_reordered = zscore_ordered.reindex(index=list(my_colors.keys()), columns=list(my_colors.keys()))

# %%
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(
    zscore_reordered.values,
    cmap='inferno',
    vmin=-50,
    vmax=100
)

ax.set_title('Neighborhood enrichment')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)

n_clusters = len(list(my_colors.keys()))
ax.set_xticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_yticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_xticklabels(list(my_colors.keys()), rotation=90)
ax.set_yticklabels(list(my_colors.keys()))

cbar = plt.colorbar(heatmap, shrink=0.7, aspect=30)
cbar.set_label('')

plt.tight_layout()

plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/NIH5900BA17/NIH5900BA17_nhood_enrich.svg', format='svg')
plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/NIH5900BA17/NIH5900BA17_nhood_enrich.pdf', format='pdf')

plt.show()

# %%
ad = sq.read.vizgen(path='./',
    counts_file="./UCSF2018003MFG/cell_by_gene.csv",
    meta_file="./UCSF2018003MFG/cell_metadata.csv")

annot = pd.read_csv('./UCSF2018003MFG/UCSF2018003MFG_annotation.csv',
                   sep=',', index_col=0)

annot = annot[['type']]

ad = ad[annot.index,]

ad.obs = ad.obs.join(annot)

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, inplace=True)
sc.pp.log1p(ad)
sc.pp.pca(ad)
sc.pp.neighbors(ad)

ad.obs['Cluster'] = ad.obs['type']

# %%
sc.pl.embedding(ad, basis="spatial", color='type', palette={k: v for k, v in my_colors.items() if k in ad.obs.type.value_counts().index})

# %%
sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key="spatial", radius=65)
sq.gr.nhood_enrichment(ad, cluster_key="type")

# %%
sq.pl.nhood_enrichment(
    ad,
    cluster_key="type",
    method=None,
    cmap="inferno",
    vmin=-50,
    vmax=100,
    figsize=(8, 8))

# %%
old_order = sorted(list(my_colors.keys()))

zscore_data = ad.uns['type_nhood_enrichment']['zscore']
zscore_ordered = pd.DataFrame(zscore_data, columns=old_order, index=old_order)

zscore_reordered = zscore_ordered.reindex(index=list(my_colors.keys()), columns=list(my_colors.keys()))

# %%
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(
    zscore_reordered.values,
    cmap='inferno',
    vmin=-50,
    vmax=100
)

ax.set_title('Neighborhood enrichment')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)

n_clusters = len(list(my_colors.keys()))
ax.set_xticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_yticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_xticklabels(list(my_colors.keys()), rotation=90)
ax.set_yticklabels(list(my_colors.keys()))

cbar = plt.colorbar(heatmap, shrink=0.7, aspect=30)
cbar.set_label('')

plt.tight_layout()

plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/UCSF2018003MFG/UCSF2018003MFG_nhood_enrich.svg', format='svg')
plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/UCSF2018003MFG/UCSF2018003MFG_nhood_enrich.pdf', format='pdf')

plt.show()

# %%
ad = sq.read.vizgen(path='./',
    counts_file="./NIH4392BA17/cell_by_gene.csv",
    meta_file="./NIH4392BA17/cell_metadata.csv")

annot = pd.read_csv('./NIH4392BA17/NIH4392BA17_annotation.csv',
                   sep=',', index_col=0)

annot = annot[['type']]

ad = ad[annot.index.astype(str),]
annot.index = annot.index.astype(str)

ad.obs = ad.obs.join(annot)

ad.layers["counts"] = ad.X.copy()
sc.pp.normalize_total(ad, inplace=True)
sc.pp.log1p(ad)
sc.pp.pca(ad)
sc.pp.neighbors(ad)

ad.obs['Cluster'] = ad.obs['type']

# %%
sc.pl.embedding(ad, basis="spatial", color='type', palette={k: v for k, v in my_colors.items() if k in ad.obs.type.value_counts().index})

# %%
sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key="spatial", radius=65)
sq.gr.nhood_enrichment(ad, cluster_key="type")

# %%
sq.pl.nhood_enrichment(
    ad,
    cluster_key="type",
    method=None,
    cmap="inferno",
    vmin=-50,
    vmax=100,
    figsize=(8, 8))

# %%
old_order=list(pd.Categorical(ad.obs['type']).categories)
zscore_data = ad.uns['type_nhood_enrichment']['zscore']
zscore_ordered = pd.DataFrame(zscore_data, columns=old_order, index=old_order)

# %%
zscore_ordered.loc[len(zscore_ordered)] = -50

# %%
zscore_ordered['IPC-EN'] = -50

# %%
new_index = list(zscore_ordered.index)
new_index = new_index[:-1] # remove 28
new_index.append("IPC-EN")

# %%
zscore_ordered.index = new_index

# %%
zscore_reordered = zscore_ordered.reindex(index=list(my_colors.keys()), columns=list(my_colors.keys()))

# %%
fig, ax = plt.subplots(figsize=(12, 12))
heatmap = ax.imshow(
    zscore_reordered.values,
    cmap='inferno',
    vmin=-50,
    vmax=100
)

ax.set_title('Neighborhood enrichment')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)

n_clusters = len(list(my_colors.keys()))
ax.set_xticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_yticks(np.linspace(0, n_clusters - 1, n_clusters))
ax.set_xticklabels(list(my_colors.keys()), rotation=90)
ax.set_yticklabels(list(my_colors.keys()))

cbar = plt.colorbar(heatmap, shrink=0.7, aspect=30)
cbar.set_label('')

plt.tight_layout()

plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/NIH4392BA17/NIH4392BA17_nhood_enrich_2.svg', format='svg')
plt.savefig('/c4/home/juanmoriano/P2_spatial_transcriptomics/231015_data/NIH4392BA17/NIH4392BA17_nhood_enrich_2.pdf', format='pdf')

plt.show()
