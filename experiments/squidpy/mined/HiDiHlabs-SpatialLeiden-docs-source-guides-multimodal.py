# mined from: https://github.com/HiDiHlabs/SpatialLeiden/blob/efca7b66f37fb395435ebfed3c9cdffdd8a6d23b/docs/source/guides/multimodal.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# %%
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from mudata import MuData

from spatialleiden import spatialleiden_multimodal

# settings
simulation = Path("../path/to/data") / "Dataset13_Simulation1"

seed = 42  # random seed for reproducibility

# %%
modalities = {}

# load modalities
for file in simulation.glob("*.h5ad"):
    modality = file.stem.split("_")[1]
    adata = ad.read_h5ad(file)
    # only keep the raw counts
    adata.X = adata.layers["counts"]
    del adata.layers, adata.uns, adata.varm
    modalities[modality] = adata

# %%
# prepare MuData
mdata = MuData(modalities)
mdata.var_names_make_unique()

mdata.obs["groundtruth"] = pd.Categorical(
    mdata.mod["RNA"].obsm["spfac"].astype(int) @ np.array([1, 2, 3, 4])
)

mdata.obsm["spatial"] = mdata["RNA"].obsm["spatial"]

for _, mod in mdata.mod.items():
    del mod.obsm

print(mdata)

# %%
# process modalities
for name, adata in mdata.mod.items():
    n_pcs = 30 if name == "RNA" else 10

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=n_pcs, random_state=seed)

# %%
# generate kNN graph per modality
for adata in mdata.mod.values():
    sc.pp.neighbors(adata, use_rep="X_pca", random_state=seed)

# %%
# spatial neighbors
sq.gr.spatial_neighbors(mdata, coord_type="grid", n_neighs=8)

# %%
# cluster with spatialleiden
spatialleiden_multimodal(
    mdata,
    resolution=0.5,
    layer_weights={"ADT": 1, "RNA": 1.5, "spatial": 2.5},
    random_state=seed,
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

scatter_kwargs = dict(x="x", y="y", s=25, lw=0, legend=False)


def remove_tick_and_label(ax):
    ax.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    ax.tick_params(left=False, bottom=False)


labels = mdata.obs[["groundtruth", "spatialleiden"]]
labels[["x", "y"]] = mdata.obsm["spatial"]

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(6, 3))

for label, ax in zip(["groundtruth", "spatialleiden"], axs):
    _ = sns.scatterplot(data=labels, hue=label, ax=ax, **scatter_kwargs)
    ax.set(title=label, aspect=1)
    remove_tick_and_label(ax)

fig.tight_layout()
