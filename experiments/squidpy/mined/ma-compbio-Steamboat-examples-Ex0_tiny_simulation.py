# mined from: https://github.com/ma-compbio/Steamboat/blob/b9c1192f10232147b4b5db54f89d558b85a22367/examples/Ex0_tiny_simulation.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
from functools import reduce
import pickle as pkl

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics

# %%
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'arial'
matplotlib.rc('pdf', fonttype=42)

do_savefig = False
savefig_path = './'

# %%
np.random.seed(42)

## Create a grid
width = 50
height = 8
xs = np.arange(0, width)
ys = np.arange(0, height)
grid_x, grid_y = np.meshgrid(xs, ys)
grid_x = grid_x.ravel()
grid_y = grid_y.ravel()
df = pd.DataFrame(np.array([grid_x, grid_y]).T, 
                  columns=['x', 'y'], 
                  index=reduce(np.char.add, ['cell_', grid_x.astype(str), '_', grid_y.astype(str)]))

## Randomly assign cell types
for y in range(3, 8):
    random_columns = np.random.choice([0, 1, 2], width, p=[0.3, 0.3, 0.4])
    
    df.loc[(random_columns[df['x']] == 0) & (df['y'] == y), 'celltype'] = 'B'
    df.loc[(random_columns[df['x']] == 1) & (df['y'] == y), 'celltype'] = 'C'
    df.loc[(random_columns[df['x']] == 2) & (df['y'] == y), 'celltype'] = 'D'

df.loc[(df['y'] <= 2), 'celltype'] = 'A'

## Generate receptor expression
def go(d, x, y, width, height):
    if d == 'left':
        x = x - 1
    elif d == 'right':
        x = x + 1
    elif d == 'up':
        y = y + 1
    elif d == 'down':
        y = y - 1
    elif d == 'left-up':
        x = x - 1
        y = y + 1
    elif d == 'right-up':
        x = x + 1
        y = y + 1
    elif d == 'left-down':
        x = x - 1
        y = y - 1 
    elif d == 'right-down':
        x = x + 1
        y = y - 1
    
    if x < 0 or y < 0 or x >= width or y >= height:
        return None, None
    else:
        return x, y

df['R+'] = 0
for i in df.index:
    if df.loc[i, 'celltype'] in ['B', 'C']:
        for d in ['left', 'right', 'up', 'down', 'left-up', 'right-up', 'left-down', 'right-down']:
            x, y = go(d, df.loc[i, 'x'], df.loc[i, 'y'], width, height)
            if x is not None and y is not None:
                j = f'cell_{x}_{y}'
                if df.loc[j, 'celltype'] == 'A':
                    if np.random.rand() < 1.:
                        df.loc[i, 'R+'] = 1

meta_expr_split = df.shape[1] # Number of columns for metadata, the rest is "expression"

# Housekeeping genes that are uniform over all cell types
df['H1'] = 50
df['H2'] = 50
df['H3'] = 50
df['H4'] = 50

# Cell identity genes that are over expressed in the corresponding cell type
df['A1'] = 1
df['A2'] = 1
df['B1'] = 1
df['B2'] = 1
df['C1'] = 1
df['C2'] = 1
df['D1'] = 1
df['D2'] = 1

df.loc[df['celltype'] == 'A', 'A1'] = 50 
df.loc[df['celltype'] == 'A', 'A2'] = 50
df.loc[df['celltype'] == 'B', 'B1'] = 50
df.loc[df['celltype'] == 'B', 'B2'] = 50
df.loc[df['celltype'] == 'C', 'C1'] = 50
df.loc[df['celltype'] == 'C', 'C2'] = 50
df.loc[df['celltype'] == 'D', 'D1'] = 50
df.loc[df['celltype'] == 'D', 'D2'] = 50

# LR
df['R'] = 1
df.loc[:, 'R'] = 100 * df['R+']

features = df.columns[meta_expr_split:].tolist()
print(*features)

expr = np.random.poisson(df[features])
expr = pd.DataFrame(expr, index=df.index, columns=features)
expr[['H1', 'H2', 'H3', 'H4']] = 50

# %%
meta = df.iloc[:, :meta_expr_split]

# %%
import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt

## Add path to the directory containing steamboat.
sys.path.append("../") 

import torch
import pandas as pd
import numpy as np
import scipy as sp
import scanpy as sc
import squidpy as sq
import steamboat as sf # Steamboat Factorization -> sf
import steamboat.tools

# %%
device = "cuda"

# %%
adata = sc.AnnData(expr, obs=meta)
adata.obsm['spatial'] = np.array(meta[['x', 'y']])
adata.obs['global'] = 0
adata

adatas = [adata] # You can include multiple datasets here.
adatas = sf.prep_adatas(adatas)
dataset = sf.make_dataset(adatas, regional_obs=[])

# %%
sq.pl.spatial_scatter(adatas[0], color=adatas[0].var_names, shape=None, figsize=(2, 1), size=1., 
                      legend_fontsize=9, cmap='Reds', ncols=4)

# %%
sq.pl.spatial_scatter(adatas[0], color='celltype', shape=None, figsize=(3, 2), size=1., 
                      legend_fontsize=9, cmap='Reds', ncols=4)

# %%
sq.pl.spatial_scatter(adatas[0], color='R', shape=None, figsize=(3, 2), size=1., 
                      legend_fontsize=9, cmap='Reds', ncols=4)

# %%
sf.set_random_seed(2)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=5, n_scales=2)
model = model.to(device)

model.fit(dataset, entry_masking_rate=0.2, feature_masking_rate=0.2,
          max_epoch=10000, 
          loss_fun=torch.nn.MSELoss(reduction='sum'),
          opt=torch.optim.Adam, opt_args=dict(lr=0.01), stop_eps=1e-3, report_per=200, stop_tol=1000, device=device)

# %%
sf.tools.plot_all_transforms2(model, top=0)

# %%
# Calculate the embedding, graph, and reconstructed cells (if needed) and store them in obs/obsm/uns
sf.tools.calculate_obs(adatas, dataset, model, get_recon=False)

# For multiple slides, use the following function to gather all obs/obsm/uns of individual slides into the whole AnnData
# sf.tools.gather_obs(adata, adatas)

# %%
i = 0

quantile = 1 - 0.01

ego = np.quantile(adatas[i].obsm['ego_attn'], quantile, axis=0)
local = np.quantile(adatas[i].obsm['local_attn'], quantile, axis=0)
# regional = np.quantile(adatas[i].obsm['regional_attn_0'], quantile, axis=0)

fig, ax = plt.subplots(figsize=(4, 1))
sns.heatmap(np.vstack([ego, local]) * 100, vmax=50, ax=ax, linewidths=0.2, linecolor='grey', cmap='Reds', annot=True, fmt='.0f')
ax.set_yticklabels(['ego', 'local'], rotation=0)

fig, ax = plt.subplots(figsize=(4, 1))
matrix = np.vstack([ego, local])
matrix /= matrix.sum(axis=0)
sns.heatmap((matrix * 100).round(), vmax=100, ax=ax, linewidths=0.2, linecolor='grey', cmap='Reds', annot=True, fmt='.0f')
ax.set_yticklabels(['ego', 'local'], rotation=0)

# %%
sf.tools.neighbors(adata, 'attn')

# %%
sf.tools.leiden(adata, resolution=0.1)

# %%
sq.pl.spatial_scatter(adata, color='steamboat_clusters', shape=None, figsize=(3, 2), size=1., 
                      legend_fontsize=9, cmap='Reds', ncols=4)
if do_savefig:
    plt.savefig(savefig_path + 'sim-clusters.pdf')

# %%
labels=['A', 'B', 'C', 'D', '0', '1', '2', '3', '4', '5']

df = pd.DataFrame(sklearn.metrics.confusion_matrix(adata.obs['celltype'], adata.obs['steamboat_clusters'], labels=labels),
            columns=labels, index=labels).loc[['A', 'B', 'C', 'D'], ['0', '1', '2', '3', '4', '5']]
df /= df.sum(axis=0)
fig, ax = plt.subplots(figsize=(1.6, 1.0))
sns.heatmap(df, linewidths=.5, ax=ax, cmap='Reds', square=True)
ax.set_xlabel('Clustering')
ax.set_ylabel('Cell type')
if do_savefig:
    plt.savefig(savefig_path + 'sim-clusters-conf.pdf')

# %%
sf.tools.segment(adata, resolution=0.33)

# %%
sq.pl.spatial_scatter(adata, color='steamboat_spatial_domain', shape=None, figsize=(3, 2), size=1., 
                      legend_fontsize=9, cmap='Reds', ncols=4)
if do_savefig:
    plt.savefig(savefig_path + 'sim-sd.pdf')

# %%
labels=['A', 'B', 'C', 'D', '0', '1']

df = pd.DataFrame(sklearn.metrics.confusion_matrix(adata.obs['celltype'], adata.obs['steamboat_spatial_domain'], labels=labels),
            columns=labels, index=labels).loc[['A', 'B', 'C', 'D'], ['0', '1']]
df = df.div(df.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(.7, 1.0))
sns.heatmap(df, linewidths=.5, ax=ax, cmap='Reds', square=True)
ax.set_xlabel('Spatial domain')
ax.set_ylabel('Cell type')
plt.savefig('sim-sd-conf.pdf')
