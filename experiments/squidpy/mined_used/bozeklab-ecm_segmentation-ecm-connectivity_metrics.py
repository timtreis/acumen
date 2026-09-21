# mined from: https://github.com/bozeklab/ecm_segmentation/blob/cf404a32ab7ce27738813698aea570b688fc176e/ecm/connectivity_metrics.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.interaction_matrix, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from cellpose import denoise, io
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pickle
import csv
import cv2
from sklearn.cluster import MiniBatchKMeans
import scanpy as sc
import spatialleiden as sl
import squidpy as sq
import anndata as ad

# %%
# Read CSVs
df_intensity_norm = pd.read_csv("intensities_normalized.csv", delimiter=",", quotechar='|')

# --- Normalized intensities --- #
# With centroids
df_intensity_norm_w_centroids = df_intensity_norm[
    ['filename', 'centroid-0', 'centroid-1'] +
    [c for c in df_intensity_norm.columns if c.endswith('normalized')]
]

# Without centroids
df_intensity_norm = df_intensity_norm[
    [c for c in df_intensity_norm.columns if c.endswith('normalized')]
]

# %%
numeric_df = df_intensity_norm.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# %%
#### KMEANS
scores = {}
#for i in range(2,9):
kmeans = MiniBatchKMeans(
    n_clusters=9,
    batch_size=16384,
    max_iter=200,
    n_init=3,
    random_state=42
)
labels = kmeans.fit_predict(X_scaled)
df_intensity_norm_w_centroids['cluster'] = labels
df_intensity_norm['cluster'] = labels

score = silhouette_score(
    X_scaled,
    labels,
    sample_size=5000,   # even 5000 often fine
    random_state=42
)
print(
"For n_clusters =",
f'{9}',
"The average silhouette_score is :",
score,
)

# %%
X = df_intensity_norm_w_centroids.drop(columns=['centroid-0', 'centroid-1', 'cluster']).values

# Coordinates
coords = df_intensity_norm_w_centroids[['centroid-0', 'centroid-1']].to_numpy()

# Cluster labels (as categorical)
df_intensity_norm_w_centroids['cluster'] = labels.astype(str)

# Create AnnData
adata = ad.AnnData(X=X)
adata.obs_names = df_intensity_norm_w_centroids.index.astype(str)

# Add obs columns (metadata)
for col in df_intensity_norm_w_centroids.columns:
    if col not in ['cluster', 'centroid-0', 'centroid-1']:
        adata.obs[col] = df_intensity_norm_w_centroids[col].values

# Add clusters
adata.obs['cluster'] = pd.Categorical(labels.astype(str))
print(adata.obs['cluster'])

# Add spatial coordinates
adata.obsm['spatial'] = coords

# Optionally, you can store feature names
adata.var_names = df_intensity_norm_w_centroids.drop(columns=['centroid-0','centroid-1','cluster']).columns

# %%
sq.gr.spatial_neighbors(
    adata, 
    coord_type='generic',
    n_neighs=10
)

print(adata)

# %%
subset_idx = np.append(0, adata.obsp['spatial_connectivities'][0, :].nonzero())

adata_sub = adata[subset_idx, :].copy()
adata_sub.obs['cluster'] = pd.Series(adata_sub.obs['cluster'], index=adata_sub.obs_names, dtype='category')


sq.pl.spatial_scatter(
    adata_sub,
    color='cluster',     
    connectivity_key='spatial_connectivities',
    shape=None,
    img=False,
    size=200,
    alpha=0.8)

# %%
file = "TMA11_43"
adata_file = adata[adata.obs['filename'] == file].copy()

sq.gr.spatial_neighbors(adata_file, spatial_key="spatial")

sq.pl.spatial_scatter(
    adata_file,
    color="cluster",
    connectivity_key="spatial_connectivities",
    shape=None,
    img=False,
    size=20,
)

# %%
###Ripleys statistics, cooccurence, interaction matrix 
sq.gr.interaction_matrix(adata, cluster_key="cluster")
sq.pl.interaction_matrix(adata, cluster_key="cluster", cmap = "plasma")

# %%
sq.gr.nhood_enrichment(adata, cluster_key="cluster")
sq.pl.nhood_enrichment(adata, cluster_key="cluster", cmap = "plasma")

# %%
for file in adata.obs['filename'].unique():
    print(file)
    adata_file = adata[adata.obs['filename'] == file]
    sq.gr.co_occurrence(adata_file, cluster_key="cluster", show_progress_bar = True, interval = 100)
    sq.pl.co_occurrence(adata_file, cluster_key="cluster", palette = 'tab10')
    plt.show()

# %%
sq.pl.co_occurrence(adata, cluster_key="cluster")

# %%
sq.pl.spatial_scatter(adata, color="cluster", size=10, shape=None)

# %%


# %%


# %%

