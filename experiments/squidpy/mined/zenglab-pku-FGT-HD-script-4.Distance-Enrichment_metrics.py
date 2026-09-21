# mined from: https://github.com/zenglab-pku/FGT-HD/blob/6646b2e38020eeed2ea0650cccd22596baafc4f7/script/4.Distance/Enrichment_metrics.ipynb
# symbols: squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import seaborn as sns
from scipy.spatial import cKDTree

import os
import json
import gc
import warnings
from tqdm import tqdm

import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
warnings.filterwarnings("ignore")

# %%
n_rings = 3
n_neighs = 8

# %%
sample_list = pd.read_csv("HD-OV 100.csv")
path = "../out"
figure_output = "../figures"

output_path = f'{path}/neighgor_count'
figure_path = f'{figure_output}/neighgor_count'

os.makedirs(output_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

# %%
with open("./ref/cell_markers.json", "r") as json_file:
    markers = json.load(json_file)

order = list(markers.keys())
order.append("Others")

# %%
def likelihood_cal(nhood_enrichment_count):
    new_df = pd.DataFrame(np.zeros_like(nhood_enrichment_count), columns=nhood_enrichment_count.columns, index=nhood_enrichment_count.index)

    Nt = nhood_enrichment_count.values.sum()
    Ni = nhood_enrichment_count.sum(axis=1)

    for i in range(len(new_df)):
        for j in range(len(new_df.columns)):
            Nij = nhood_enrichment_count.iloc[i, j]
            Nj = nhood_enrichment_count.iloc[:, j].sum()
            new_df.iloc[i, j] = Nij * Nt / (Ni[i] * Nj)

    log_df = np.log2(new_df)
    return log_df

# %%

adata = sc.read_h5ad(f"HGSOC_adata.h5ad")
adata.obs['in_tissue'] = adata.obs['in_tissue'].astype(float)
adata.obs['array_row'] = adata.obs['array_row'].astype(float)
adata.obs['array_col'] = adata.obs['array_col'].astype(float)
adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
adata.obs['annotations'] = pd.Categorical(adata.obs['annotations'], categories=order, ordered=True)

sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid", n_neighs=n_neighs)
sq.gr.nhood_enrichment(adata, cluster_key="annotations", n_perms=10)
sq_df_neibor = pd.DataFrame(adata.uns["annotations_nhood_enrichment"]['count'], index=order, columns=order)
sq_df_neibor.to_csv(f"{output_path}/neighbor_count.csv")

# score_with_others = likelihood_cal(sq_df_neibor)

sq_df_neibor_wo = sq_df_neibor.drop(index="Others", errors="ignore")
sq_df_neibor_wo = sq_df_neibor_wo.drop(columns="Others", errors="ignore")
score_wo_others = likelihood_cal(sq_df_neibor_wo)

abs_data = score_wo_others.abs()
max_abs_val = abs_data.max().max()
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#483d8b", "#ffffff", "#800000"])

sns.heatmap(
    score_wo_others,
    cmap=custom_cmap,
    center=0,
    vmin=-max_abs_val, vmax=max_abs_val,
    annot=True, fmt=".2f",
    linewidths=0.5
)

plt.savefig(f"{figure_path}/neighbor_likelihood.pdf", format="pdf", bbox_inches='tight')
plt.close()

# %%
def calculate_distances(obs, cell_type_1, cell_type_2):
    results = []
    for sample, group in obs.groupby('sample'):
        type_1_coords = group[group['annotations'] == cell_type_1][['array_row', 'array_col']].values
        type_2_coords = group[group['annotations'] == cell_type_2][['array_row', 'array_col']].values

        if len(type_1_coords) > 0 and len(type_2_coords) > 0:
            tree = cKDTree(type_1_coords)
            if cell_type_1 != cell_type_2:
                distances, indices = tree.query(type_2_coords, k=1)
            else:
                distances, indices = tree.query(type_2_coords, k=2)
                distances = distances[:,1]

            results.append((sample, distances))
        
    result_df = pd.DataFrame(results, columns=['Sample', 'Distances'])

    return result_df

# %%
adata = sc.read_h5ad("HGSOC_adata.h5ad")
adata.obs['in_tissue'] = adata.obs['in_tissue'].astype(float)
adata.obs['array_row'] = adata.obs['array_row'].astype(float)
adata.obs['array_col'] = adata.obs['array_col'].astype(float)
adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
adata.obs['annotations'] = pd.Categorical(adata.obs['annotations'], categories=order, ordered=True)

obs = adata.obs

# %%
T2cancer_distance_df = calculate_distances(obs, "T", "Epithelial")
T2cancer_dis = np.concatenate(T2cancer_distance_df['Distances'].values)

# %%

