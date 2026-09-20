# mined from: https://github.com/ma-compbio/Steamboat/blob/b9c1192f10232147b4b5db54f89d558b85a22367/reproducibility/Ex1_hgsc/fine_domains.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
import scanpy as sc
import squidpy as sq
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

# %%
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'arial'

pltkw = dict(bbox_inches='tight', transparent=True)

# %%
import sys
sys.path.append("..")
import steamboat as sf
import steamboat.tools
import torch
device = 'cuda'

# %%
import importlib
importlib.reload(steamboat.tools)

# %%
# https://www.nature.com/articles/s41590-024-01943-5

# %%
regenerate = False

h5ad_file = "../../../data/HGSC/ST_Discovery_so.h5ad"
if (not os.path.exists(h5ad_file)) or regenerate:
    adata = sc.read_mtx("G:/data/HGSC/Csv/ST_Discovery_so_counts.mtx").T
    metadata = pd.read_csv("G:/data/HGSC/Csv/ST_Discovery_so_metadata.csv", index_col=0)
    features = pd.read_csv("G:/data/HGSC/Csv/ST_Discovery_so_features.txt", index_col=0, header=None)
    features.index = features.index.str.strip() # remove trailing white space in gene names
    features.index.name = 'gene_symbol'
    adata.obs = metadata
    adata.var = features
    adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()
    adata.write_h5ad("G:/data/HGSC/h5ad/ST_Discovery_so.h5ad")
else:
    adata = sc.read_h5ad(h5ad_file)

# %%
adata.obs

# %%
TNK_info = adata.obs[adata.obs['cell.types'] == 'Monocyte']
TNK_info['cell.subtypes'].value_counts()

# %%
## Metadata and gene sets

sample_metadata = pd.read_excel("../../../data/HGSC/sample_metadata.xlsx", index_col=0, sheet_name='Table 2b', skiprows=1)
sample_metadata = sample_metadata[sample_metadata['dataset'] == 'Discovery']

celltype_signatures = pd.read_excel("../../../data/HGSC/sample_metadata.xlsx", sheet_name='Table 3a', skiprows=2)
mtil_signautures = pd.read_excel("../../../data/HGSC/sample_metadata.xlsx", sheet_name='Table 6a', skiprows=2)
desmoplasia_signautures = pd.read_excel("../../../data/HGSC/sample_metadata.xlsx", sheet_name='Table 5a', skiprows=2)

def purge_gene_sets(df, prefix=''):
    res = {}
    for i in df.columns:
        res[prefix + i] = df[i].dropna().tolist()
    return res
celltype_signatures = pd.read_excel("../../../data/HGSC/sample_metadata.xlsx", index_col=0, sheet_name='Table 3b', skiprows=2).iloc[:, :-3]
genesets = (purge_gene_sets(celltype_signatures, 'sig_') | 
            purge_gene_sets(mtil_signautures, 'mtil_') | 
            purge_gene_sets(mtil_signautures, 'mtil_'))
genesets.keys()
del genesets['sig_Mast.cell']

sample_metadata

# %%
## Find untreated, adnexa samples

columns_of_interest = ['sites_binary', 'stage', 'treatment']
fig, axes = plt.subplots(1, len(columns_of_interest), figsize=(len(columns_of_interest) * 1.5, 3))
for i, column in enumerate(columns_of_interest):
    sample_metadata[column].value_counts().plot(kind='bar', ax=axes[i])
    axes[i].set_title(column)
plt.tight_layout()

mask = (sample_metadata['sites_binary'] == 'Adnexa') & (sample_metadata['treatment'] == 'Untreated')
samples_of_interest = sample_metadata.index[mask].tolist()

all_adata = adata[adata.obs['samples'].isin(samples_of_interest)].copy()
all_adata.obs['cell.types.nolc'] = all_adata.obs['cell.types'].str.replace('_LC', '')

# %%
# selected_samples = np.random.choice(all_adata.obs['samples'].unique(), size=10, replace=False)
# all_adata = all_adata[all_adata.obs['samples'].isin(selected_samples)].copy()

# %%


# %%
# Separate individual slides
adatas = []
for i in all_adata.obs['samples'].unique():
    temp = all_adata[all_adata.obs['samples'] == i].copy()
    if temp.shape[0] < 100:
        continue
    adatas.append(temp)
    adatas[-1].obs['global'] = 0

# normalize and log transformation
adatas = sf.prep_adatas(adatas, norm=True, log1p=True, scale=False, renorm=False)

# create torch dataset
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

# %%
print(*adata.obs['cell.types'].unique())

# %%
immune_types = ['TNK.cell', 'B.cell', 'Monocyte', 'Mast.cell']
fibroblast_types = ['Fibroblast']

# %%
cuda_dataset = None

load_data_into_gpu = True # if you run into OOM on GPU, set this to False
if device == 'cuda' and load_data_into_gpu:
    cuda_dataset = dataset.to('cuda')

# %%
n_heads = 25

sf.set_random_seed(0)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=n_heads, n_scales=3)
model = model.to(device)

use_dataset = cuda_dataset
if use_dataset is None:
    use_dataset = dataset

model.load_state_dict(torch.load('../examples/saved_models/hgsc.pth', weights_only=True), strict=False)

# model.fit(cuda_dataset, entry_masking_rate=0.1, feature_masking_rate=0.1,
#           max_epoch=10000, 
#           loss_fun=torch.nn.MSELoss(reduction='sum'),
#           opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=1e-3, report_per=200, stop_tol=200, device=device)

# %%
sf.tools.calc_obs(adatas, dataset, model, get_recon=True)

# %%
adatas[0].obs

# %%
for i in range(len(adatas)):
    adata = adatas[i]
    if 'steamboat_spatial_domain_colors' in adata.uns:
        adata.uns.pop('steamboat_spatial_domain_colors')
    sf.tools.segment(adata, resolution=.7, key_added="steamboat_spatial_domain", n_prop=2)
    
    pd.crosstab(adata.obs['steamboat_spatial_domain'], adata.obs['cell.types.nolc']).to_csv(f"./saved_results/hgsc_fine_spatial_domain/hgsc_{adata.obs['samples'][0]}_steamboat_spatial_domain_crosstab.csv")
    adata.obs['steamboat_spatial_domain'].to_csv(f"./saved_results/hgsc_fine_spatial_domain/hgsc_{adata.obs['samples'][0]}_steamboat_spatial_domain.csv")
    
    sq.pl.spatial_scatter(adata, color=["steamboat_spatial_domain", "cell.types.nolc"], size=.1, shape=None, legend_loc='right margin', frameon=False, figsize=(3, 3), ncols=1)
    plt.savefig(f"./saved_results/hgsc_fine_spatial_domain/hgsc_{adata.obs['samples'][0]}_steamboat_spatial_domain.png", bbox_inches='tight')
    break

# %%
df = pd.crosstab(adatas[0].obs['steamboat_spatial_domain'], adatas[0].obs['cell.types.nolc'])
n = df.loc['0', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
N = n + df.loc['0', 'Malignant']
m = df.loc['1', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
M = m + df.loc['1', 'Malignant']

# proportion_test(N, n, M, m)
df / df.sum(axis=0)

# %%
df
