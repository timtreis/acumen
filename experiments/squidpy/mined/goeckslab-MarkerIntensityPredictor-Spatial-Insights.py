# mined from: https://github.com/goeckslab/MarkerIntensityPredictor/blob/b4596f39f0eae52cb8325f96a0e5c14ccf621d2f/Spatial Insights.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
biopsy = pd.read_csv("data/tumor_mesmer/9_2_1.csv")

# %%
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']
SPATIAL_INFORMATION = ["X_centroid","Y_centroid"]

# %%
biopsy = biopsy[SHARED_MARKERS + SPATIAL_INFORMATION]
#predictions = 

# %%
biopsy.describe()

# %%
biopsy.head()

# %%
import squidpy as sq
import anndata as ad
import numpy as np
from pathlib import Path

# %%


# %%
coordinates = biopsy[["X_centroid", "Y_centroid"]]
coordinates = coordinates.to_numpy()
coordinates = coordinates.reshape(-1,2)
coordinates

# %%
adata = ad.AnnData(biopsy[SHARED_MARKERS],  dtype=np.int64)
adata

# %%
adata.obsm["spatial"] =  coordinates

# %%
results = []
radii = [15,30,60,90,120]
for radius in radii:
    neighbors = sq.gr.spatial_neighbors(adata, coord_type = 'generic', radius=radius, spatial_key="spatial")
    result = sq.gr.spatial_autocorr(adata, copy=True)
    result["Radius"] = radius
    results.append(result)
results = pd.concat(results)

# %%
results = results.reset_index()
results.rename(columns={"index": "Marker"}, inplace=True)
results

# %%


# %%
biopsies = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
radii = [15,30,60,90,120]
results = []
for biopsy in biopsies:
    biopsy_df = pd.read_csv(f"data/tumor_mesmer/{biopsy}.csv")
    biopsy_df = biopsy_df[SHARED_MARKERS + SPATIAL_INFORMATION]
    coordinates = biopsy_df[["X_centroid", "Y_centroid"]]
    coordinates = coordinates.to_numpy()
    coordinates = coordinates.reshape(-1,2)
    adata = ad.AnnData(biopsy_df[SHARED_MARKERS], obsm={"spatial": coordinates},dtype=np.int64)


    for radius in radii:
        neighbors = sq.gr.spatial_neighbors(adata, coord_type = 'generic', radius=radius, spatial_key="spatial")
        result = sq.gr.spatial_autocorr(adata, copy=True)
        result["FE"] = radius
        result["Biopsy"] = biopsy
        results.append(result)


results = pd.concat(results)

results = results.reset_index()
results.rename(columns={"index": "Marker"}, inplace=True)
results

# %%
save_path = Path("data", "cleaned_data", "spatial_information")

if not save_path.exists():
    save_path.mkdir(parents=True)

results.to_csv(Path(save_path, "spatial_clustering.csv"), index=False)

# %%

