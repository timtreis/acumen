# mined from: https://github.com/bozeklab/ecm_segmentation/blob/cf404a32ab7ce27738813698aea570b688fc176e/ecm/Leiden.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.interaction_matrix, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import spatialleiden as sl
import squidpy as sq

# %%
df = pd.read_csv("intensities_normalized.csv", delimiter=",", quotechar='|')
print(len(df['filename'].unique()))

numeric_df = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# %%
feature_cols = [c for c in numeric_df.columns if c not in ['centroid-1', 'centroid-0']]
feature_df = df[feature_cols].values

coords = df[['centroid-0', 'centroid-1']]
adata = sc.AnnData(feature_df)
adata.var_names = feature_cols
adata.obsm['spatial'] = coords.values

random_state = 42
sc.pp.scale(adata, max_value=10)
#sc.tl.pca(adata, random_state=random_state)
sc.pp.neighbors(adata, random_state=random_state)

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=20)
adata.obsp["spatial_connectivities"] = sl.distance2connectivity(
    adata.obsp["spatial_distances"]
)

# %%
random_state = 42

sc.tl.leiden(adata, flavor = 'leidenalg', directed=False, n_iterations=2, random_state=random_state, resolution = 0.2)
#sl.spatialleiden(adata, layer_ratio=1.8, directed=(False, True), random_state=random_state)
sc.settings.figdir = "images"
sc.pl.embedding(adata, basis = 'spatial', color=["leiden"], save= "_leiden_clusters_norm.png")

# Cluster labels
#adata.obs['spatialleiden']
adata.obs['leiden']

# Spatial coordinates
adata.obsm['spatial']

# %%
sc.tl.umap(adata, min_dist = 0.5)  
sc.pl.umap(adata, color = 'leiden', save= "_leiden_clusters_norm.png")

# %%
cluster_counts = adata.obs['leiden'].value_counts()
print(cluster_counts)
cluster_density = cluster_counts / len(adata)
print(cluster_density)

cluster_counts = adata.obs['leiden'].value_counts()
cluster_density = cluster_counts / len(adata)

sns.barplot(x=cluster_density.index, y=cluster_density.values)

# %%
adata.obs["filename"] = df["filename"].values
print(adata.obs["filename"].nunique())
adata.obs[["filename", "leiden"]].head()
adata.write("adata_with_leiden_norm.h5ad", compression="gzip")

# %%
adata = sc.read_h5ad("adata_with_leiden_norm.h5ad")

# %%
import matplotlib.pyplot as plt
import numpy as np

samples = adata.obs["filename"].unique()

n_cols = 10
n_rows = int(np.ceil(len(samples) / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(2 * n_cols, 2 * n_rows),
    squeeze=False
)

for i, fname in enumerate(samples):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]

    adata_sub = adata[adata.obs["filename"] == fname]

    xy = adata_sub.obsm["spatial"]
    clusters = adata_sub.obs["leiden"].astype(int)

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=clusters,
        s=2,
        cmap="tab20",
        linewidths=0
    )

    ax.set_title(fname, fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

# Hide unused axes
for j in range(i + 1, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis("off")

plt.tight_layout()
plt.show()
plt.close()

# %%
adata.uns['spatial'] = {
    'generic': {
        'images': {},
        'scalefactors': {'tissue_hires_scalef': 1.0}
    }
}
print(adata)

# %%
sq.gr.spatial_neighbors(
    adata,
    coord_type="generic",
    n_neighs=20,
    key_added="spatial"
)

# %%
# Visualize the whole core with spatial neighbors
sq.pl.spatial_scatter(
    adata,
    color="cluster",
    coord_type="generic",             
    size=50,
    img=False                          
)

# %%
adata_sub = adata[adata.obs['filename'] == 'TMA10_35'].copy()

# %%
cell_idx = 14
_, idx = adata_sub.obsp["spatial_connectivities"][cell_idx, :].nonzero()
idx = np.append(idx, cell_idx)
sq.pl.spatial_scatter(adata_sub, color="cluster", connectivity_key="spatial_connectivities", img=False, na_color="lightgrey", size = 50)

# %%
sq.pl.spatial_scatter(
    adata[idx, :],
    shape=None,
    color="leiden",
    connectivity_key="spatial_connectivities",
    size=200,
)

# %%
sq.pl.nhood_enrichment(
    adata,
    cluster_key="cluster",
    figsize=(4, 4),
    cmap="bwr"
)

# %%
sq.gr.interaction_matrix(adata, cluster_key="cluster")
sq.pl.interaction_matrix(adata, cluster_key="cluster",figsize=(4, 4),
    cmap="bwr")

# %%
# Dont know yet what it means but lets computer as many things as possible
sq.gr.co_occurrence(
    adata,
    cluster_key="cluster",
    #n_samp=1000,   # number of random permutations / samples
    #max_dist=200,  # set according to your pixel / micron scale
)

sq.pl.co_occurrence(
    adata,
    cluster_key="cluster",
   # clusters=["0", "1", "2"],  # subset if desired
   # figsize=(5, 15)
)
