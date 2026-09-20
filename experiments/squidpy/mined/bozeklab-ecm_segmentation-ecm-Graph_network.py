# mined from: https://github.com/bozeklab/ecm_segmentation/blob/cf404a32ab7ce27738813698aea570b688fc176e/ecm/Graph_network.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.interaction_matrix, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.interaction_matrix, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import uniform_filter
from skimage import filters
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
import squidpy as sq

# %%
# Loop over files in folder
folder = '/projects/ag-bozek/mcarval3/ecm/data/interim/aligned/'

for file_path in glob.glob(f"{folder}/*.npy"):
        print(file_path)

# Split filenames
_, tail = os.path.split(file_path) 
filename = os.path.splitext(tail)[0]

# %%
# Load array
array = np.load(file_path)

cycle, channel, x,y =array.shape

# %%
plt.imshow(array[0,0,:,:])

# %%
from cellpose import models, io
from cellpose.io import imread

# %%
DAPI = array[0,0, :,:] # nuclear mask first channel
model = models.CellposeModel(gpu=True)#,model_type='nuclei')
masks, flows, styles, = model.eval(DAPI, diameter=20)

# %%
plt.imshow(flows[0])

# %%
# Generate dataframes with region prop tables and combine all for all channels per cycle and then all cycle altogether
from skimage.measure import regionprops_table

cycle_dfs = {}

for cycle in range(array.shape[0]):
    all_props = []
    for channel in range(array.shape[1]):
        img = array[cycle, channel, :, :]

        props = regionprops_table(
            masks,
            intensity_image=img,
            properties=[
                'centroid',
                'orientation',
                'axis_major_length',
                'axis_minor_length',
                'label',
                'area',
                'mean_intensity',
                'max_intensity',
                'min_intensity',
            ],
        )

        df = pd.DataFrame(props)
        df['cycle'] = cycle
        df['channel'] = channel
        all_props.append(df)

    # dataframe with all channels for this cycle
    cycle_dfs[cycle] = pd.concat(all_props, ignore_index=True)

#combine all cycles into one dataframe 
all_df = pd.concat(cycle_dfs.values(), ignore_index=True)


# Multiple properties with MultiIndex columns (like in a excel file) 
pivot_multi = all_df.pivot_table(
    index='label',
    columns=['cycle', 'channel'],   # multi-level columns
    values=['mean_intensity', 'max_intensity', 'min_intensity'],
)

# %%
# Flat MultiIndex so that is a more convetional dataframe
pivot_multi.columns = [
    f'{prop}_c{c}_ch{ch}'
    for prop, c, ch in pivot_multi.columns.to_flat_index()
]

# %%
pivot_multi.columns # check all columns to see if only intensities are present 

# %%
# Extract centroids for spatial coordinates
props = regionprops_table(masks, properties=['centroid'])
coords = np.column_stack([props['centroid-0'], props['centroid-1']])

# Prepare feature matrix for clustering
features_df = pivot_multi
X = features_df.values

# Perform KMeans clustering
n_clusters = 9 #like we decided before
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(X)

# Store cluster labels and features in AnnData
features_df["cluster"] = cluster_labels.astype(str)
adata = ad.AnnData(X=X)
adata.obs_names = features_df.index.astype(str)
for col in features_df.columns:
    if col == "cluster":
        continue
    adata.obs[col] = features_df[col].values
adata.obsm["spatial"] = coords
adata.obs["cluster"] = features_df["cluster"].values

# %%
plt.figure(figsize=((x/y)*9, 8))
scatter = plt.scatter(coords[:, 1], coords[:, 0], c=cluster_labels, cmap='tab10', s=50)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spatial Plot of Clusters')
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.show()

# %%
# Local neighboorhod
idx = np.append(0, adata.obsp["spatial_connectivities"][0, :].nonzero()[1])  # Example: visualize neighbors for a given cell
sq.pl.spatial_scatter(adata[idx, :],color="cluster", connectivity_key="spatial_connectivities", img=False, na_color="black", size=50)

# %%
# Compute spatial neighbors graph for the whole core
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=10)

# Visualize the whole core with spatial neighbors
sq.pl.spatial_scatter(adata, color="cluster", connectivity_key="spatial_connectivities", img=False, na_color="lightgrey", size =50)

# %%
# neihbourhood analysis and plotting
sq.gr.nhood_enrichment(adata, cluster_key="cluster")

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

# %%

