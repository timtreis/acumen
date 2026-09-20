# mined from: https://github.com/HuangLab-PKU/PRISM-Analysis/blob/3b10d6100e489782a2375c6229ccc420ec45d655/HCC2D/nhood_enrichment.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.nhood_enrichment

# %%
# Loading the Packages
# %reload_ext autoreload
# %autoreload 2

# basic
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
from tqdm import tqdm

# calculation
import numpy as np
import pandas as pd
import scanpy as sc

# visualization
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

# %%
# workdir 
BASE_DIR = Path(r'G:\spatial_data\processed')
RUN_ID = '20230523_HCC_PRISM_probe_refined'
src_dir = BASE_DIR / f'{RUN_ID}_processed'

# Load one slide exp
base_path = BASE_DIR / f'{RUN_ID}_processed'
data_path = base_path / "segmented"
typ_path = base_path / "cell_typing"
output_path = base_path / "nhood_enrichment"
output_path.mkdir(exist_ok=True)

# %%
combine_adata_st = sc.read_h5ad(os.path.join(typ_path, 'combine_adata_st.h5ad'))
combine_adata_st = combine_adata_st[combine_adata_st.obs.type != 'other']

# %%
import yaml
with open(os.path.join(output_path, 'params.yaml')) as file:
    config = yaml.full_load(file)
type_reorder = config['type_reorder']
subtype_reorder = config['subtype_reorder']

print('type')
print('not in:', set(type_reorder)-set(combine_adata_st.obs.type.unique()))
print('not plot:', set(combine_adata_st.obs.type.unique())-set(type_reorder))

print('subtype')
print('not in:', set(subtype_reorder)-set(combine_adata_st.obs.subtype.unique()))
print('not plot:', set(combine_adata_st.obs.subtype.unique())-set(subtype_reorder))

# %%
import squidpy as sq

combine_adata_st.obsm['spatial'] = np.array([combine_adata_st.obs.X_pos,combine_adata_st.obs.Y_pos]).T
sq.gr.spatial_neighbors(combine_adata_st, n_neighs=30, coord_type="generic", spatial_key="spatial")

# %%
combine_adata_st.obs.type = pd.Categorical(combine_adata_st.obs.type, categories=type_reorder)
sq.gr.nhood_enrichment(combine_adata_st, cluster_key="type", n_perms=3000)
combine_adata_st.obs.subtype = pd.Categorical(combine_adata_st.obs.subtype, categories=subtype_reorder)
sq.gr.nhood_enrichment(combine_adata_st, cluster_key="subtype", n_perms=3000)

# %%
fig, ax = plt.subplots(figsize=(23, 10),ncols=2, nrows=1)

ax_tmp = ax[0]
enrichment_matrix = pd.DataFrame(combine_adata_st.uns["type_nhood_enrichment"]['zscore'], index=type_reorder, columns=type_reorder)
sns.heatmap(enrichment_matrix, cmap="coolwarm", vmin=-50, vmax=50, ax=ax_tmp)

ax_tmp = ax[1]
enrichment_matrix = pd.DataFrame(combine_adata_st.uns["subtype_nhood_enrichment"]['zscore'], index=subtype_reorder, columns=subtype_reorder)
sorted_matrix = enrichment_matrix.loc[subtype_reorder, subtype_reorder]
sns.heatmap(sorted_matrix, cmap="coolwarm", vmin=-50, vmax=50, ax=ax_tmp)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'nhood_enrichment.png'))
plt.close()

# %%
from skimage import io, transform, morphology

def ROI_mask_load(input_path, out_path, show=True, save=False):
    ROI_mask = {}
    for mask_file in os.listdir(input_path):
        image = io.imread(os.path.join(input_path, mask_file))
        image = transform.rotate(image, angle=90, resize=True)
        image = morphology.binary_dilation(image, footprint=morphology.disk(5))
        ROI_mask[mask_file.replace('.tif','').replace('Mask', 'ROI')] = image
        
    ncols = int(-(-len(ROI_mask)**(1/2)//1))
    nrows = -(-len(ROI_mask)//ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))
    for pos, mask_name in enumerate(list(ROI_mask.keys())):
        ax[pos // ncols][pos % ncols].imshow(ROI_mask[mask_name], cmap='gray')
        ax[pos // ncols][pos % ncols].set_title(mask_name)
        ax[pos // ncols][pos % ncols].set_xlabel("")
        ax[pos // ncols][pos % ncols].set_ylabel("")
    fig.suptitle('Mask_of_ROIs', fontsize=20)
    plt.tight_layout()

    if save: plt.savefig(out_path)
    elif show: plt.show()
    plt.close()

    return ROI_mask

# %%
ROI_mask = ROI_mask_load(input_path=os.path.join(base_path, 'roi_variation', 'roi_mask'), 
                         show=False, save=False, out_path=os.path.join(output_path, 'roi_mask.png'))

# %%
from collections import Counter


combine_adata_st.obs['ROI'] = pd.Categorical(['other']*len(combine_adata_st), categories=list(ROI_mask.keys()) + ['other'], ordered=False)    
for _, mask in ROI_mask.items():
    yrange = mask.shape[0]
    for cell in tqdm(combine_adata_st.obs.index, desc=_):
        if mask[yrange - int(combine_adata_st.obs['Y_pos'].loc[cell]/100), int(combine_adata_st.obs['X_pos'].loc[cell]/100)]:
            combine_adata_st.obs['ROI'].loc[cell] = _

# %%
# show the cell in different ROI in one plot with different color
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.spatial(combine_adata_st, color='ROI', ax=ax, show=False, spot_size=100)
# reverse y axis
plt.gca().invert_yaxis()
plt.savefig(output_path / 'ROI_projection.png')
plt.close()

# %%
plot_range = {
    'ROI_1': [-10, 10],
    'ROI_2': [-10, 10],
    'ROI_3': [-15, 15],
    'ROI_4': [-20, 20],
    'ROI_5': [-10, 10],
    'other': [-50, 50],
    }
for roi in combine_adata_st.obs.ROI.unique():
    adata_roi = combine_adata_st[combine_adata_st.obs.ROI == roi]
    vmin, vmax = plot_range[roi]
    # if file exist
    if os.path.exists(os.path.join(output_path, f'nhood_enrichment_{roi}_type.csv')):
        enrichment_matrix_type = pd.read_csv(os.path.join(output_path, f'nhood_enrichment_{roi}_type.csv'), index_col=0)
    else:
        sq.gr.nhood_enrichment(adata_roi, cluster_key="type", n_perms=3000)
        order = [_ for _ in type_reorder if _ in adata_roi.obs.type.unique()]
        enrichment_matrix_type = pd.DataFrame(adata_roi.uns["type_nhood_enrichment"]['zscore'], index=order, columns=order)
        enrichment_matrix_type.to_csv(os.path.join(output_path, f'nhood_enrichment_{roi}_type.csv'))
    
    if os.path.exists(os.path.join(output_path, f'nhood_enrichment_{roi}_subtype.csv')):
        enrichment_matrix_subtype = pd.read_csv(os.path.join(output_path, f'nhood_enrichment_{roi}_subtype.csv'), index_col=0)
    else:
        sq.gr.nhood_enrichment(adata_roi, cluster_key="subtype", n_perms=3000)
        order = [_ for _ in subtype_reorder if _ in adata_roi.obs.subtype.unique()]
        enrichment_matrix_subtype = pd.DataFrame(adata_roi.uns["subtype_nhood_enrichment"]['zscore'], index=order, columns=order)
        enrichment_matrix_subtype.to_csv(os.path.join(output_path, f'nhood_enrichment_{roi}_subtype.csv'))

    fig, ax = plt.subplots(figsize=(23, 10),ncols=2, nrows=1)
    sns.heatmap(enrichment_matrix_type, cmap="coolwarm", ax=ax[0], vmin=vmin, vmax=vmax) #, robust=True, vmin=-50, vmax=50)
    sns.heatmap(enrichment_matrix_subtype, cmap="coolwarm", ax=ax[1], vmin=vmin, vmax=vmax) #, robust=True, vmin=-50, vmax=50)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'nhood_enrichment{roi}.png'))
    plt.close()

# %%
plot_range = {
    'ROI_1': [-10, 10],
    'ROI_2': [-10, 10],
    'ROI_3': [-15, 15],
    'ROI_4': [-20, 20],
    'ROI_5': [-10, 10],
    'other': [-50, 50],
    }
for roi in combine_adata_st.obs.ROI.unique():
    adata_roi = combine_adata_st[combine_adata_st.obs.ROI == roi]
    vmin, vmax = plot_range[roi]
    sq.gr.nhood_enrichment(adata_roi, cluster_key="type", n_perms=3000)
    sq.gr.nhood_enrichment(adata_roi, cluster_key="subtype", n_perms=3000)

    fig, ax = plt.subplots(figsize=(23, 10),ncols=2, nrows=1)
    sq.pl.nhood_enrichment(adata_roi, cluster_key="type", method='ward', cmap="coolwarm", vmin=vmin, vmax=vmax, ax=ax[0])
    sq.pl.nhood_enrichment(adata_roi, cluster_key="subtype", method='ward', cmap="coolwarm", vmin=vmin, vmax=vmax, ax=ax[1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'nhood_enrichment_cluster_{roi}.png'))
    plt.close()
