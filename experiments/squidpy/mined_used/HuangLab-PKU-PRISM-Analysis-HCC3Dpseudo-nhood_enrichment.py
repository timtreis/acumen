# mined from: https://github.com/HuangLab-PKU/PRISM-Analysis/blob/3b10d6100e489782a2375c6229ccc420ec45d655/HCC3Dpseudo/nhood_enrichment.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment

# %%
# Loading the Packages
# %reload_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "xelatex",      # 使用 XeLaTeX，如果不需要 LaTeX 公式渲染，可以省略
    'font.family': 'serif',          # 字体设置为衬线字体
    'text.usetex': False,            # 禁用 LaTeX，使用 Matplotlib 内置文字渲染
    'pgf.rcfonts': False,            # 禁用 pgf 的默认字体管理
    'pdf.fonttype': 42,              # 确保字体为 TrueType 格式，可被 Illustrator 编辑
    'ps.fonttype': 42,               # EPS 文件也使用 TrueType 格式
    'figure.dpi': 300,               # 设置图形分辨率
    'savefig.dpi': 300,              # 保存的图形文件分辨率
    'axes.unicode_minus': False,     # 避免负号问题
})

# workdir 
BASE_DIR = Path(r'G:\spatial_data\analysis')
RUN_ID = '20250222_combined_analysis_of_pseudo_HCC3D'

# Load one slide exp
base_path = BASE_DIR / f'{RUN_ID}'
data_path = base_path / "segmented"
typ_path = base_path / "cell_typing"
output_path = base_path / "nhood_enrichment"
output_path.mkdir(parents=True, exist_ok=True)

# %%
combine_adata_st = sc.read_h5ad(typ_path / 'combine_adata_st.h5ad')
combine_adata_st.obsm['spatial'] = np.array([combine_adata_st.obs.X_pos,combine_adata_st.obs.Y_pos]).T
combine_adata_st.obsm['spatial3d'] = np.array([
    combine_adata_st.obs.X_pos,combine_adata_st.obs.Y_pos,
    [int(_.replace('slice',''))*10/0.1625 for _ in combine_adata_st.obs.slice]]).T
combine_adata_st = combine_adata_st[combine_adata_st.obs.type!='other']
print(combine_adata_st)
combine_adata_st.obs.head()

# %%
import yaml
with open(base_path / 'nhood_enrichment_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

type_reorder = params['type_reorder']
subtype_reorder = params['subtype_reorder']

print('type')
print('not in:', set(type_reorder)-set(combine_adata_st.obs.type.unique()))
print('not plot:', set(combine_adata_st.obs.type.unique())-set(type_reorder))

print('subtype')
print('not in:', set(subtype_reorder)-set(combine_adata_st.obs.subtype.unique()))
print('not plot:', set(combine_adata_st.obs.subtype.unique())-set(subtype_reorder))

# %%
import squidpy as sq

sq.gr.spatial_neighbors(combine_adata_st, coord_type="generic", spatial_key="spatial3d")

# %%
combine_adata_st.obs.type = pd.Categorical(combine_adata_st.obs.type, categories=type_reorder)
sq.gr.nhood_enrichment(combine_adata_st, cluster_key="type", n_perms=3000)
combine_adata_st.obs.subtype = pd.Categorical(combine_adata_st.obs.subtype, categories=subtype_reorder)
sq.gr.nhood_enrichment(combine_adata_st, cluster_key="subtype", n_perms=3000)

# %%
fig,ax = plt.subplots(figsize=(20,10), ncols=2, nrows=1)
sq.pl.nhood_enrichment(combine_adata_st, cluster_key="type", method='ward', cmap="coolwarm", ax=ax[0], vmin=-100, vmax=100)
sq.pl.nhood_enrichment(combine_adata_st, cluster_key="subtype", method='ward', cmap="coolwarm", ax=ax[1], vmin=-80, vmax=80)
plt.tight_layout()
plt.show()

# %%
import scipy.cluster.hierarchy as sch

fig, ax = plt.subplots(figsize=(23, 10),ncols=2, nrows=1)
order = type_reorder
enrichment_matrix = pd.DataFrame(combine_adata_st.uns["type_nhood_enrichment"]['zscore'], index=order, columns=order)
# linkage = sch.linkage(enrichment_matrix, method='ward')
# dendrogram = sch.dendrogram(linkage, no_plot=True)
# order = [int(i) for i in dendrogram['leaves']]
# sorted_matrix = enrichment_matrix.iloc[order, order]
sns.heatmap(enrichment_matrix, cmap="coolwarm", vmin=-80, vmax=80, ax=ax[0])

order = subtype_reorder
enrichment_matrix = pd.DataFrame(combine_adata_st.uns["subtype_nhood_enrichment"]['zscore'], index=order, columns=order)
# linkage = sch.linkage(enrichment_matrix, method='ward')
# dendrogram = sch.dendrogram(linkage, no_plot=True)
# order = [int(i) for i in dendrogram['leaves']]
# sorted_matrix = enrichment_matrix.iloc[order, order]
sns.heatmap(enrichment_matrix, cmap="coolwarm", vmin=-80, vmax=80, ax=ax[1])

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'nhood_enrichment.png'))
plt.close()

# %%
# show the cell in different ROI in one plot with different color
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.spatial(combine_adata_st[combine_adata_st.obs.slice=='slice10'], color='region', ax=ax, show=False, spot_size=100)
# reverse y axis
plt.gca().invert_yaxis()
plt.savefig(output_path / 'region_projection.png')
plt.close()

# %%
plot_range = {
    'ROI1': [-50, 50],
    'ROI2': [-40, 40],
    'ROI3': [-50, 50],
    'ROI4': [-50, 50],
    'other': [-50, 50],
    }
for roi in combine_adata_st.obs.region.unique():
    adata_roi = combine_adata_st[combine_adata_st.obs.region == roi]
    vmin, vmax = plot_range[roi]
    # vmin, vmax = None, None
    sq.gr.nhood_enrichment(adata_roi, cluster_key="type", n_perms=3000)
    sq.gr.nhood_enrichment(adata_roi, cluster_key="subtype", n_perms=3000)

    fig, ax = plt.subplots(figsize=(23, 10),ncols=2, nrows=1)
    sq.pl.nhood_enrichment(adata_roi, cluster_key="type", method='ward', cmap="coolwarm", vmin=vmin, vmax=vmax, ax=ax[0])
    sq.pl.nhood_enrichment(adata_roi, cluster_key="subtype", method='ward', cmap="coolwarm", vmin=vmin, vmax=vmax, ax=ax[1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'nhood_enrichment_cluster_region_{roi}.png'))
    plt.close()

# %%
# show the cell in different ROI in one plot with different color
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.spatial(combine_adata_st[combine_adata_st.obs.slice=='slice10'], color='ROI', ax=ax, show=False, spot_size=100)
# reverse y axis
plt.gca().invert_yaxis()
plt.savefig(output_path / 'ROI_projection.png')
plt.close()

# %%
plot_range = {
    'ROI_1': [-20, 20],
    'ROI_2': [-10, 10],
    'ROI_3': [-15, 15],
    'ROI_4': [-25, 25],
    'ROI_5': [-20, 20],
    'other': [-100, 200],
    }
for roi in combine_adata_st.obs.ROI.unique():
    adata_roi = combine_adata_st[combine_adata_st.obs.ROI == roi]
    vmin, vmax = plot_range[roi]
    # vmin, vmax = None, None
    sq.gr.nhood_enrichment(adata_roi, cluster_key="type", n_perms=3000)
    sq.gr.nhood_enrichment(adata_roi, cluster_key="subtype", n_perms=3000)

    fig, ax = plt.subplots(figsize=(23, 10),ncols=2, nrows=1)
    sq.pl.nhood_enrichment(adata_roi, cluster_key="type", method='ward', cmap="coolwarm", vmin=vmin, vmax=vmax, ax=ax[0])
    sq.pl.nhood_enrichment(adata_roi, cluster_key="subtype", method='ward', cmap="coolwarm", vmin=vmin, vmax=vmax, ax=ax[1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'nhood_enrichment_cluster_ROI_{roi}.png'))
    plt.close()
